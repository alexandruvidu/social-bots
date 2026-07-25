"""Flask webhook server for Instagram official Messaging API.

Run: python -m bot.webhook

Routes:
  GET  /webhook  — Meta verification handshake
  POST /webhook  — Incoming DM events
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import sys
import time
from threading import Thread

from flask import Flask, abort, request

from .config import Config
from .run import drain_retry_queue, handle_posts, handle_replies
from .sources.instagram_api import InstagramAPISource
from .store import Store

log = logging.getLogger("bot.webhook")
app = Flask(__name__)

# How often the background thread checks for rate-limited posts that are due
# for a retry. Runs in its own thread with its own Store/source so it never
# shares a sqlite connection or instagrapi session with request handling.
RETRY_POLL_INTERVAL = 45.0

# Lazily initialised on first request so tests can swap out _build_components.
_components: tuple | None = None


def _build_components() -> tuple[Config, InstagramAPISource, Store]:
    from .sources.instagram import InstagramSource

    cfg = Config.load()
    enrich = InstagramSource(
        username=cfg.ig_username,
        password=cfg.ig_password,
        allowed_sender_id=cfg.allowed_sender_id,
        session_path=cfg.session_path,
        comments_limit=cfg.comments_limit,
        comments_fetch_limit=cfg.comments_fetch_limit,
    )
    source = InstagramAPISource(
        access_token=cfg.ig_access_token or "",
        ig_user_id=cfg.ig_user_id or "",
        allowed_sender_id=cfg.allowed_sender_id,
        enrich_source=enrich,
    )
    store = Store(cfg.db_path)
    return cfg, source, store


def _get_components() -> tuple[Config, InstagramAPISource, Store]:
    global _components
    if _components is None:
        _components = _build_components()
    return _components


def verify_signature(payload: bytes, signature_header: str | None, app_secret: str) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        log.warning("Missing or malformed X-Hub-Signature-256 header: %r", signature_header)
        return False
    expected = hmac.new(app_secret.encode(), payload, hashlib.sha256).hexdigest()
    received = signature_header[7:]
    if not hmac.compare_digest(received, expected):
        log.warning(
            "Signature mismatch: received=%s... expected=%s... secret_len=%d payload_len=%d",
            received[:8], expected[:8], len(app_secret), len(payload),
        )
        return False
    return True


@app.route("/webhook", methods=["GET"])
def webhook_verify():
    cfg, _, _ = _get_components()
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge", "")
    if mode == "subscribe" and token == cfg.webhook_verify_token:
        return challenge, 200
    abort(403)


@app.route("/webhook", methods=["POST"])
def webhook_event():
    cfg, source, store = _get_components()
    sig = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(request.data, sig, cfg.ig_app_secret or ""):
        abort(403)

    data = request.get_json(force=True) or {}
    if data.get("object") != "instagram":
        return "ok", 200

    posts = []
    replies = []

    for entry in data.get("entry", []):
        for messaging in entry.get("messaging", []):
            sender_id = str(messaging.get("sender", {}).get("id", ""))
            log.info("Incoming event from sender %s", sender_id)
            if sender_id == str(cfg.ig_user_id):
                continue  # own echo
            if sender_id != str(cfg.allowed_sender_id):
                log.info("Ignoring untrusted sender %s (expected %s)", sender_id, cfg.allowed_sender_id)
                continue
            message = messaging.get("message", {})
            if message.get("attachments"):
                post = source.build_post_from_event(messaging)
                if post:
                    posts.append(post)
                else:
                    log.info("Attachment message from %s did not yield a post.", sender_id)
            elif message.get("text"):
                reply = source.build_reply_from_event(messaging)
                if reply:
                    replies.append(reply)
                else:
                    log.info("Text message from %s did not yield a reply.", sender_id)
            else:
                log.info("Message from %s had neither attachments nor text: %r", sender_id, message)

    log.info("Webhook batch: %d post(s), %d reply(ies) to process.", len(posts), len(replies))
    handle_replies(store, source, replies)
    handle_posts(store, source, posts, cfg)
    return "ok", 200


def _retry_loop() -> None:
    cfg = Config.load()
    store = Store(cfg.db_path)
    source = InstagramAPISource(
        access_token=cfg.ig_access_token or "",
        ig_user_id=cfg.ig_user_id or "",
        allowed_sender_id=cfg.allowed_sender_id,
    )
    while True:
        time.sleep(RETRY_POLL_INTERVAL)
        try:
            drain_retry_queue(store, source, cfg)
        except Exception:
            log.exception("Retry queue drain failed")


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("bot").setLevel(logging.INFO)
    Thread(target=_retry_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=False)


if __name__ == "__main__":
    main()
