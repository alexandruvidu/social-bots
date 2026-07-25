"""Webhook and InstagramAPISource tests — all offline."""
import os
import pytest


def test_config_loads_webhook_vars(monkeypatch):
    monkeypatch.setenv("IG_USERNAME", "u")
    monkeypatch.setenv("IG_PASSWORD", "p")
    monkeypatch.setenv("ALLOWED_SENDER_ID", "111")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("IG_ACCESS_TOKEN", "tok123")
    monkeypatch.setenv("IG_APP_SECRET", "sec456")
    monkeypatch.setenv("WEBHOOK_VERIFY_TOKEN", "mytoken")
    monkeypatch.setenv("IG_USER_ID", "999")

    from bot.config import Config
    cfg = Config.load()
    assert cfg.ig_access_token == "tok123"
    assert cfg.ig_app_secret == "sec456"
    assert cfg.webhook_verify_token == "mytoken"
    assert cfg.ig_user_id == "999"


def test_config_webhook_vars_default_none(monkeypatch):
    monkeypatch.setenv("IG_USERNAME", "u")
    monkeypatch.setenv("IG_PASSWORD", "p")
    monkeypatch.setenv("ALLOWED_SENDER_ID", "111")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    for var in ("IG_ACCESS_TOKEN", "IG_APP_SECRET", "WEBHOOK_VERIFY_TOKEN", "IG_USER_ID"):
        monkeypatch.delenv(var, raising=False)

    from bot.config import Config
    cfg = Config.load()
    assert cfg.ig_access_token is None
    assert cfg.ig_app_secret is None
    assert cfg.webhook_verify_token is None
    assert cfg.ig_user_id is None


def test_config_boots_without_ig_credentials(monkeypatch):
    """The whole point of the change: no password on disk, still starts."""
    monkeypatch.setenv("ALLOWED_SENDER_ID", "111")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("IG_USERNAME", raising=False)
    monkeypatch.delenv("IG_PASSWORD", raising=False)

    from bot.config import Config
    cfg = Config.load()
    assert cfg.ig_username is None
    assert cfg.ig_password is None


def test_build_components_does_not_construct_instagram_source(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOWED_SENDER_ID", "111")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("IG_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("IG_USER_ID", "999")
    # _build_components opens a real sqlite file — keep it out of data/.
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.sqlite"))

    from unittest.mock import patch
    import bot.webhook as wh

    with patch("bot.sources.instagram.InstagramSource") as mock_src:
        cfg, source, store = wh._build_components()

    mock_src.assert_not_called()
    assert not hasattr(source, "_enrich")


def test_attachment_replying_to_pending_ask_routes_to_media_handler():
    from unittest.mock import MagicMock, patch
    import json as _json
    import bot.webhook as wh

    cfg = MagicMock()
    cfg.ig_user_id = "999"
    cfg.allowed_sender_id = "111"
    source = MagicMock()
    source.platform = "instagram"
    store = MagicMock()
    store.get_pending_by_ask_msg.return_value = {"link": "L", "caption_snippet": None,
                                                 "ask_msg_id": "ask_1"}
    wh._components = (cfg, source, store)

    body = {"object": "instagram", "entry": [{"messaging": [{
        "sender": {"id": "111"},
        "message": {"mid": "m2", "reply_to": {"mid": "ask_1"},
                    "attachments": [{"type": "image",
                                     "payload": {"url": "https://lookaside.fbsbx.com/s"}}]},
    }]}]}

    with patch("bot.webhook.verify_signature", return_value=True), \
         patch("bot.webhook.handle_posts") as posts, \
         patch("bot.webhook.handle_replies"), \
         patch("bot.webhook.handle_media_replies") as media:
        wh.app.test_client().post("/webhook", data=_json.dumps(body),
                                  content_type="application/json")

    wh._components = None
    source.build_media_reply_from_event.assert_called_once()
    source.build_post_from_event.assert_not_called()
    assert media.call_args.args[2] != []
    assert posts.call_args.args[2] == []


def test_attachment_without_pending_ask_routes_to_post_handler():
    from unittest.mock import MagicMock, patch
    import json as _json
    import bot.webhook as wh

    cfg = MagicMock()
    cfg.ig_user_id = "999"
    cfg.allowed_sender_id = "111"
    source = MagicMock()
    source.platform = "instagram"
    store = MagicMock()
    store.get_pending_by_ask_msg.return_value = None
    wh._components = (cfg, source, store)

    body = {"object": "instagram", "entry": [{"messaging": [{
        "sender": {"id": "111"},
        "message": {"mid": "m1",
                    "attachments": [{"type": "ig_reel",
                                     "payload": {"url": "https://www.instagram.com/reel/X/"}}]},
    }]}]}

    with patch("bot.webhook.verify_signature", return_value=True), \
         patch("bot.webhook.handle_posts") as posts, \
         patch("bot.webhook.handle_replies"), \
         patch("bot.webhook.handle_media_replies") as media:
        wh.app.test_client().post("/webhook", data=_json.dumps(body),
                                  content_type="application/json")

    wh._components = None
    source.build_post_from_event.assert_called_once()
    source.build_media_reply_from_event.assert_not_called()
    assert media.call_args.args[2] == []


def test_enrich_from_url_returns_post_with_link_on_pk_failure():
    from unittest.mock import MagicMock, patch
    from bot.sources.instagram import InstagramSource

    src = MagicMock()
    src.client = MagicMock()
    src.client.media_pk_from_url.side_effect = Exception("nope")

    # Call the real method with the mock as self
    from bot.sources.instagram import InstagramSource as IS
    result = IS.enrich_from_url(src, "mid1", "sender1", "https://www.instagram.com/reel/ABC/")

    from bot.sources.base import SharedPost
    assert isinstance(result, SharedPost)
    assert result.link == "https://www.instagram.com/reel/ABC/"
    assert result.item_id == "mid1"
    assert result.thread_id == "sender1"
    assert result.caption is None


def test_enrich_from_url_calls_build_post_on_success():
    from unittest.mock import MagicMock, patch
    from bot.sources.instagram import InstagramSource
    from bot.sources.base import SharedPost

    src = MagicMock()
    src.client = MagicMock()
    src.client.media_pk_from_url.return_value = "12345"
    expected = SharedPost(
        platform="instagram", item_id="mid1", thread_id="sender1",
        link="https://www.instagram.com/reel/ABC/", caption="nice view"
    )
    src._build_post.return_value = expected

    from bot.sources.instagram import InstagramSource as IS
    result = IS.enrich_from_url(src, "mid1", "sender1", "https://www.instagram.com/reel/ABC/")

    assert result is expected
    src._build_post.assert_called_once()
    call_args = src._build_post.call_args
    assert call_args.args[0] == "mid1"
    assert call_args.args[1] == "sender1"
    assert call_args.args[2].pk == "12345"


# ── HMAC verification ──────────────────────────────────────────────────────

def test_verify_signature_valid():
    import hashlib, hmac as _hmac
    from bot.webhook import verify_signature
    payload = b'{"test": 1}'
    secret = "mysecret"
    sig = "sha256=" + _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert verify_signature(payload, sig, secret) is True


def test_verify_signature_invalid():
    from bot.webhook import verify_signature
    assert verify_signature(b"data", "sha256=wrongvalue", "secret") is False


def test_verify_signature_missing_header():
    from bot.webhook import verify_signature
    assert verify_signature(b"data", None, "secret") is False


def test_verify_signature_wrong_prefix():
    from bot.webhook import verify_signature
    assert verify_signature(b"data", "md5=abc123", "secret") is False


# ── GET /webhook verification route ───────────────────────────────────────

@pytest.fixture
def flask_client(monkeypatch):
    monkeypatch.setenv("IG_USERNAME", "u")
    monkeypatch.setenv("IG_PASSWORD", "p")
    monkeypatch.setenv("ALLOWED_SENDER_ID", "111")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("IG_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("IG_APP_SECRET", "appsecret")
    monkeypatch.setenv("WEBHOOK_VERIFY_TOKEN", "myverifytoken")
    monkeypatch.setenv("IG_USER_ID", "bot999")

    # Prevent real Instagram login during test setup
    from unittest.mock import patch as _patch, MagicMock
    with _patch("bot.webhook._build_components") as mock_build:
        from bot.config import Config
        cfg = Config.load()
        mock_source = MagicMock()
        mock_store = MagicMock()
        mock_build.return_value = (cfg, mock_source, mock_store)

        import importlib
        import bot.webhook
        importlib.reload(bot.webhook)

        bot.webhook.app.config["TESTING"] = True
        with bot.webhook.app.test_client() as client:
            yield client, cfg, mock_source, mock_store


def test_get_webhook_valid_token(flask_client):
    client, cfg, _, _ = flask_client
    resp = client.get("/webhook", query_string={
        "hub.mode": "subscribe",
        "hub.verify_token": "myverifytoken",
        "hub.challenge": "challenge_abc",
    })
    assert resp.status_code == 200
    assert resp.data == b"challenge_abc"


def test_get_webhook_wrong_token(flask_client):
    client, _, _, _ = flask_client
    resp = client.get("/webhook", query_string={
        "hub.mode": "subscribe",
        "hub.verify_token": "wrongtoken",
        "hub.challenge": "challenge_abc",
    })
    assert resp.status_code == 403


def test_get_webhook_wrong_mode(flask_client):
    client, _, _, _ = flask_client
    resp = client.get("/webhook", query_string={
        "hub.mode": "unsubscribe",
        "hub.verify_token": "myverifytoken",
        "hub.challenge": "challenge_abc",
    })
    assert resp.status_code == 403


# ── POST /webhook event handler ───────────────────────────────────────────

import hashlib, hmac as _hmac, json


def _sign(payload: bytes, secret: str = "appsecret") -> str:
    return "sha256=" + _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _post_event(client, body: dict, secret: str = "appsecret"):
    raw = json.dumps(body).encode()
    return client.post(
        "/webhook",
        data=raw,
        content_type="application/json",
        headers={"X-Hub-Signature-256": _sign(raw, secret)},
    )


def test_post_webhook_rejects_bad_signature(flask_client):
    client, _, _, _ = flask_client
    resp = client.post(
        "/webhook",
        data=b'{"object":"instagram"}',
        content_type="application/json",
        headers={"X-Hub-Signature-256": "sha256=badhash"},
    )
    assert resp.status_code == 403


def test_post_webhook_ignores_non_instagram_object(flask_client):
    client, _, mock_source, mock_store = flask_client
    body = {"object": "page", "entry": []}
    resp = _post_event(client, body)
    assert resp.status_code == 200
    mock_source.build_post_from_event.assert_not_called()


def test_post_webhook_processes_shared_reel(flask_client):
    from unittest.mock import MagicMock
    client, _, _, _ = flask_client

    # Mock handle_posts to avoid real API calls
    import bot.webhook
    original_handle_posts = bot.webhook.handle_posts
    bot.webhook.handle_posts = MagicMock()

    try:
        body = {
            "object": "instagram",
            "entry": [{
                "messaging": [{
                    "sender": {"id": "111"},
                    "message": {
                        "mid": "mid1",
                        "attachments": [{"type": "ig_reel", "payload": {"url": "https://www.instagram.com/reel/XYZ/"}}],
                    },
                }]
            }]
        }
        resp = _post_event(client, body)
        assert resp.status_code == 200
        # Verify handle_posts was called with a post (exact contents depend on real source)
        bot.webhook.handle_posts.assert_called_once()
        call_args = bot.webhook.handle_posts.call_args
        # Check that posts list has one item
        posts_list = call_args[0][2]
        assert len(posts_list) == 1
        # Check that the post has the expected item_id from the event
        assert posts_list[0].item_id == "mid1"
    finally:
        bot.webhook.handle_posts = original_handle_posts


def test_post_webhook_ignores_untrusted_sender(flask_client):
    client, _, mock_source, mock_store = flask_client
    body = {
        "object": "instagram",
        "entry": [{
            "messaging": [{
                "sender": {"id": "evil_stranger"},
                "message": {"mid": "mid1", "text": "hack"},
            }]
        }]
    }
    resp = _post_event(client, body)
    assert resp.status_code == 200
    mock_source.build_reply_from_event.assert_not_called()
    mock_source.build_post_from_event.assert_not_called()


def test_post_webhook_ignores_own_echoes(flask_client):
    client, cfg, mock_source, _ = flask_client
    body = {
        "object": "instagram",
        "entry": [{
            "messaging": [{
                "sender": {"id": "bot999"},   # same as IG_USER_ID
                "message": {"mid": "mid1", "text": "hello"},
            }]
        }]
    }
    resp = _post_event(client, body)
    assert resp.status_code == 200
    mock_source.build_reply_from_event.assert_not_called()


def test_post_webhook_processes_text_reply(flask_client):
    from unittest.mock import MagicMock
    client, _, _, _ = flask_client

    # Mock handle_replies to avoid real API calls
    import bot.webhook
    original_handle_replies = bot.webhook.handle_replies
    bot.webhook.handle_replies = MagicMock()

    try:
        body = {
            "object": "instagram",
            "entry": [{
                "messaging": [{
                    "sender": {"id": "111"},
                    "message": {"mid": "mid2", "text": "Tokyo, Japan"},
                }]
            }]
        }
        resp = _post_event(client, body)
        assert resp.status_code == 200
        # Verify handle_replies was called with a reply (exact contents depend on real source)
        bot.webhook.handle_replies.assert_called_once()
        call_args = bot.webhook.handle_replies.call_args
        # Check that replies list has one item
        replies_list = call_args[0][2]
        assert len(replies_list) == 1
        # Check that the reply has the expected item_id from the event
        assert replies_list[0].item_id == "mid2"
    finally:
        bot.webhook.handle_replies = original_handle_replies
