"""Instagram source backed by the official Messaging API (Graph API).

Receives events via webhook (bot/webhook.py) — fetch_new() is never called.
Sends replies via POST /{ig_user_id}/messages.
Uses an InstagramSource instance for media enrichment (caption/comments).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from .base import SharedPost, TextReply

if TYPE_CHECKING:
    from .instagram import InstagramSource

log = logging.getLogger(__name__)

PLATFORM = "instagram"
GRAPH_API = "https://graph.instagram.com/v21.0"
SHARE_ATTACHMENT_TYPES = {"ig_reel", "share", "video", "image"}


class InstagramAPISource:
    platform = PLATFORM

    def __init__(
        self,
        access_token: str,
        ig_user_id: str,
        allowed_sender_id: str,
        enrich_source: "InstagramSource | None" = None,
    ):
        self.access_token = access_token
        self.ig_user_id = str(ig_user_id)
        self.allowed_sender_id = str(allowed_sender_id)
        self._enrich = enrich_source

    def fetch_new(self) -> tuple[list[SharedPost], list[TextReply]]:
        return [], []

    def reply(
        self, thread_id: str, text: str, reply_to_item_id: str | None = None
    ) -> str | None:
        try:
            resp = httpx.post(
                f"{GRAPH_API}/{self.ig_user_id}/messages",
                params={"access_token": self.access_token},
                json={"recipient": {"id": thread_id}, "message": {"text": text}},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json().get("message_id")
        except httpx.HTTPStatusError as exc:
            log.warning(
                "Failed to send reply to %s: HTTP %d - %s",
                thread_id, exc.response.status_code, exc.response.text,
            )
            return None
        except Exception:
            log.warning("Failed to send reply to %s", thread_id, exc_info=True)
            return None

    def persist(self) -> None:
        pass

    def build_post_from_event(self, messaging: dict) -> SharedPost | None:
        sender_id = str(messaging["sender"]["id"])
        message = messaging.get("message", {})
        mid = message.get("mid", "")
        attachments = message.get("attachments", [])
        if not attachments:
            log.info("build_post_from_event: no attachments on message %s", mid)
            return None
        attachment = attachments[0]
        atype = attachment.get("type")
        if atype not in SHARE_ATTACHMENT_TYPES:
            log.info(
                "build_post_from_event: unsupported attachment type %r for message %s; payload=%r",
                atype, mid, attachment.get("payload"),
            )
            return None
        url = attachment.get("payload", {}).get("url", "")
        if not url:
            log.info("build_post_from_event: attachment %r has no payload.url for message %s", atype, mid)
            return None
        log.info("build_post_from_event: parsed %s attachment for message %s -> %s", atype, mid, url)
        if self._enrich:
            return self._enrich.enrich_from_url(mid, sender_id, url)
        return SharedPost(platform=PLATFORM, item_id=mid, thread_id=sender_id, link=url)

    def build_reply_from_event(self, messaging: dict) -> TextReply | None:
        sender_id = str(messaging["sender"]["id"])
        message = messaging.get("message", {})
        mid = message.get("mid", "")
        text = message.get("text", "")
        if not text:
            return None
        reply_to = message.get("reply_to") or {}
        reply_to_mid = reply_to.get("mid")
        return TextReply(
            platform=PLATFORM,
            item_id=mid,
            thread_id=sender_id,
            text=text.strip(),
            reply_to_item_id=reply_to_mid,
        )
