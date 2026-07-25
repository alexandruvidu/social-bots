"""Platform-agnostic Source interface.

Adding TikTok later means writing another class that satisfies `Source` and
returns the same `SharedPost` / `TextReply` shapes — the orchestrator and the
rest of the pipeline don't change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class PostComment:
    text: str
    is_creator: bool = False   # authored by the post's creator — the trustworthy signal
    likes: int = 0
    author: str | None = None


@dataclass
class SharedPost:
    platform: str
    item_id: str          # DM message id — used for processed-tracking
    thread_id: str
    link: str
    caption: str | None = None
    location: str | None = None
    comments: list[PostComment] = field(default_factory=list)
    media_url: str | None = None
    # "video" | "image" | None. None means the source couldn't tell (Instagram's
    # `share` attachment type covers both), so the type is sniffed from the
    # response Content-Type at download time.
    media_kind: str | None = None


@dataclass
class MediaReply:
    """An image the user sent in reply to the bot's "where is this?" ask."""
    platform: str
    item_id: str
    thread_id: str
    media_url: str
    reply_to_item_id: str | None = None  # ID of the ask message being answered


@dataclass
class TextReply:
    platform: str
    item_id: str
    thread_id: str
    text: str
    reply_to_item_id: str | None = None  # ID of the message the user replied to, if any


class Source(Protocol):
    platform: str

    def fetch_new(self) -> tuple[list[SharedPost], list[TextReply]]:
        """Return (shared posts, plain-text replies) from trusted DM threads.

        Implementations should enrich each SharedPost (caption/location/comments)
        before returning. Items already in the `processed` table may still be
        returned; the orchestrator filters them out.
        """
        ...

    def reply(self, thread_id: str, text: str, reply_to_item_id: str | None = None) -> str | None:
        """Send a DM back to a thread. Returns the sent message ID if available."""
        ...

    def persist(self) -> None:
        """Persist any session state (e.g. auth cookies). No-op if not needed."""
        ...
