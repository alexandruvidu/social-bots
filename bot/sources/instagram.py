"""Instagram Source backed by the unofficial `instagrapi` private-API client.

DORMANT. Nothing in the live pipeline imports this module: constructing this
class logs in to Instagram's private API, which is what got the bot account
forcibly logged out everywhere. It is kept on disk so the change is reversible,
and is reachable only via `bot.run` with ALLOW_PRIVATE_API=1 explicitly set.

Reads DM threads, pulls shared posts/reels from the trusted sender, enriches each
with caption + geotag + top comments, and exposes plain-text replies (used to
resolve a failed extraction).
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

from instagrapi import Client
from instagrapi.exceptions import ChallengeRequired, FeedbackRequired

from .base import PostComment, SharedPost, Source, TextReply

log = logging.getLogger(__name__)

PLATFORM = "instagram"

# Instagram's "we limit how often you can do this" / challenge response — a
# temporary, account-level action block on the private media-lookup endpoint,
# not a per-post error. Once seen, back off entirely for a while: retrying
# immediately just burns the (already-forbidden) public fallback too and
# risks turning a soft block into a longer one (see README ban-risk note).
#
# Instagram doesn't tell us how long the block lasts (no Retry-After-style
# field on the FeedbackRequired response — confirmed by inspecting
# client.last_json live). instagrapi's own docs/maintainer guidance just say
# to stop the blocked action and "give the account a rest for the day", not
# minutes, so start conservative and escalate further if it keeps recurring
# (a sign the cooldown was too short and we re-tripped it).
ACTION_BLOCK_EXCEPTIONS = (FeedbackRequired, ChallengeRequired)
ACTION_BLOCK_BASE_COOLDOWN_S = 4 * 60 * 60
ACTION_BLOCK_MAX_COOLDOWN_S = 24 * 60 * 60

_LOCATION_HINT_RE = re.compile(r"(?i)\blocation\b")
_CONNECTOR_WORDS = {"de", "da", "do", "du", "of", "van", "von", "di", "la", "le", "el", "the", "and"}


def _looks_like_place_name(text: str) -> bool:
    """Heuristic for "this comment is probably naming a place".

    Catches explicit "Location: X" callouts and runs of 2+ capitalized words
    (a likely proper noun), e.g. "Miradouro de São Cristovão" — connector
    words like "de"/"of" don't break the run.
    """
    if _LOCATION_HINT_RE.search(text):
        return True
    streak = 0
    for word in text.split():
        bare = word.strip(",.!?:;()'\"")
        if bare and bare[0].isupper() and bare.isalpha():
            streak += 1
            if streak >= 2:
                return True
        elif bare.lower() not in _CONNECTOR_WORDS:
            streak = 0
    return False


def _select_comments(comments: list[PostComment], limit: int) -> list[PostComment]:
    """Pick the `limit` most useful comments out of everything fetched.

    Instagram's default order is mostly chronological, so the actual answer
    to a "where is this?" question often lands well past the early flood of
    reaction comments (emoji, "stunning!", unanswered "where is it?"). Plain
    truncation at `limit` can bury it entirely. Always keep creator comments
    and anything that looks like a place name; fill any remaining slots by
    like count.
    """
    if len(comments) <= limit:
        return comments
    priority = [c for c in comments if c.is_creator or _looks_like_place_name(c.text)]
    rest = sorted(
        (c for c in comments if c not in priority),
        key=lambda c: c.likes,
        reverse=True,
    )
    return (priority + rest)[:limit]


def _prompt(message: str) -> str:
    """Prompt on the terminal; fail clearly if there's no TTY (e.g. under cron)."""
    if not sys.stdin.isatty():
        raise SystemExit(
            "Instagram needs an interactive verification step, but no terminal is "
            "attached. Run `python -m bot.run` (or `python -m bot.whoami <handle>`) "
            "manually once to complete login — the session is then saved to "
            "data/session.json and future runs (including cron) reuse it."
        )
    return input(message).strip()


def login_client(username: str, password: str, session_path: Path) -> Client:
    """Log in, handling saved sessions, email/SMS challenges, and 2FA.

    On the first login Instagram may send a one-time code or require a 2FA code;
    this prompts for it interactively, then the caller persists the session.
    """
    from instagrapi.exceptions import TwoFactorRequired

    client = Client()
    # Randomized per-request delay — instagrapi's own best-practices guide
    # recommends this so request timing doesn't look perfectly robotic
    # (fixed/zero delays between calls are a flagged pattern).
    client.delay_range = [1, 3]
    # Challenge (email/SMS code) handler — instagrapi calls this when prompted.
    client.challenge_code_handler = lambda u, choice: _prompt(
        f"Enter the verification code Instagram sent for {u} ({choice}): "
    )

    if session_path.exists():
        try:
            client.load_settings(session_path)
        except Exception:  # noqa: BLE001
            log.warning("Could not load saved session; logging in fresh.")

    def _do_login() -> None:
        try:
            client.login(username, password)
        except TwoFactorRequired:
            code = _prompt(f"Enter your 2FA code for {username}: ")
            client.login(username, password, verification_code=code)

    try:
        _do_login()
    except Exception:
        # A stale session can poison login — clear it and try once more fresh.
        log.warning("Login failed; clearing session and retrying fresh.")
        client.set_settings({})
        client.challenge_code_handler = lambda u, choice: _prompt(
            f"Enter the verification code Instagram sent for {u} ({choice}): "
        )
        _do_login()

    return client

# DirectMessage.item_type values that carry a shared Media, mapped to the
# attribute on the message that holds it.
SHARE_ATTRS = {
    "media_share": "media_share",
    "clip": "clip",
    "felix_share": "felix_share",
    "xma_media_share": "xma_share",
    "xma_reel_share": "xma_share",
    "xma_clip": "xma_share",
}


def _media_link(client: Client, media) -> str | None:
    # Prefer the stable instagram.com permalink over the CDN video_url, which is
    # an ephemeral signed asset URL, not a page users/the bot should link to.
    code = getattr(media, "code", None)
    if code:
        return f"https://www.instagram.com/p/{code}/"
    # xma_share objects (embedded shares) may lack `code`; fall back to their
    # ready-made CDN URL, stripped of query params to clean it up.
    video_url = getattr(media, "video_url", None)
    if video_url:
        from urllib.parse import urlparse
        p = urlparse(str(video_url))
        return f"{p.scheme}://{p.netloc}{p.path}"
    pk = getattr(media, "pk", None) or getattr(media, "id", None)
    if pk:
        try:
            return client.media_link(pk)
        except Exception:  # noqa: BLE001
            return None
    return None


def _location_str(media) -> str | None:
    loc = getattr(media, "location", None)
    if not loc:
        return None
    bits = [getattr(loc, "name", None), getattr(loc, "city", None)]
    return ", ".join(b for b in bits if b) or None


def _video_url(media) -> str | None:
    url = getattr(media, "video_url", None)
    if not url:
        return None
    return str(url)


class InstagramSource(Source):
    platform = PLATFORM

    def __init__(
        self,
        username: str | None,
        password: str | None,
        allowed_sender_id: str,
        session_path: Path,
        comments_limit: int = 25,
        comments_fetch_limit: int = 100,
        threads_amount: int = 20,
    ):
        # Config.ig_username/ig_password are optional (the live webhook path
        # never logs in), so they can legitimately be None here. Say so plainly
        # rather than handing None to client.login() and failing opaquely.
        if not username or not password:
            raise ValueError(
                "InstagramSource requires both a username and a password "
                "(IG_USERNAME / IG_PASSWORD); got username=%r, password=%s."
                % (username, "set" if password else "missing")
            )
        self.allowed_sender_id = str(allowed_sender_id)
        self.comments_limit = comments_limit
        self.comments_fetch_limit = max(comments_fetch_limit, comments_limit)
        self.threads_amount = threads_amount
        self.session_path = session_path
        self.client = login_client(username, password, session_path)
        self._msg_cache: dict[str, object] = {}  # item_id → DirectMessage
        self._media_info_blocked_until: float = 0.0
        self._action_block_streak: int = 0
        # Save immediately so a successful first login is never lost.
        self.persist()

    def persist(self) -> None:
        try:
            self.client.dump_settings(self.session_path)
        except Exception:  # noqa: BLE001
            log.warning("Failed to persist IG session.", exc_info=True)

    def reply(self, thread_id: str, text: str, reply_to_item_id: str | None = None) -> str | None:
        reply_to = self._msg_cache.get(reply_to_item_id) if reply_to_item_id else None
        sent = self.client.direct_send(text, thread_ids=[thread_id], reply_to_message=reply_to)
        return str(sent.id) if sent else None

    def fetch_new(self) -> tuple[list[SharedPost], list[TextReply]]:
        posts: list[SharedPost] = []
        replies: list[TextReply] = []

        threads = self.client.direct_threads(amount=self.threads_amount)
        for thread in threads:
            thread_id = str(thread.id)
            for msg in getattr(thread, "messages", []) or []:
                msg_user_id = str(getattr(msg, "user_id", ""))
                if msg_user_id != self.allowed_sender_id:
                    continue
                item_id = str(msg.id)
                item_type = getattr(msg, "item_type", None)
                self._msg_cache[item_id] = msg

                if item_type == "text" and getattr(msg, "text", None):
                    raw_reply = getattr(msg, "reply", None)
                    reply_to_item_id = str(raw_reply.id) if raw_reply else None
                    replies.append(
                        TextReply(
                            platform=PLATFORM,
                            item_id=item_id,
                            thread_id=thread_id,
                            text=msg.text.strip(),
                            reply_to_item_id=reply_to_item_id,
                        )
                    )
                    continue

                attr = SHARE_ATTRS.get(item_type)
                if not attr:
                    continue
                media = getattr(msg, attr, None)
                if media is None:
                    continue
                post = self._build_post(item_id, thread_id, media)
                if post:
                    posts.append(post)

        return posts, replies

    def _trip_action_block(self, pk: str, call_name: str) -> None:
        # Instagram gives no Retry-After-style hint, so escalate: each block
        # hit back-to-back (i.e. the previous cooldown was too short, or a
        # different private call tripped the same account-level block) doubles
        # the wait, up to a day.
        cooldown = min(
            ACTION_BLOCK_BASE_COOLDOWN_S * (2 ** self._action_block_streak),
            ACTION_BLOCK_MAX_COOLDOWN_S,
        )
        self._action_block_streak += 1
        self._media_info_blocked_until = time.monotonic() + cooldown
        log.warning(
            "%s action-blocked by Instagram for %s; backing off %ds (streak=%d).",
            call_name, pk, cooldown, self._action_block_streak,
        )

    def _build_post(self, item_id: str, thread_id: str, media) -> SharedPost | None:
        # Enrich via media_info for reliable caption/location; fall back to the
        # embedded media if that call fails (rate limit, private, etc.).
        pk = (
            getattr(media, "pk", None)
            or getattr(media, "id", None)
            or getattr(media, "preview_media_fbid", None)
        )
        full = media
        # media_info_v1 and media_comments are both private-API calls gated by
        # the same account-level action block — checked once so a block from
        # either one skips both for the rest of the cooldown, instead of
        # blindly retrying the comments call every post.
        blocked = bool(pk) and time.monotonic() < self._media_info_blocked_until
        if pk:
            if blocked:
                log.info("media_info skipped for %s; IG action-block cooldown active.", pk)
            else:
                try:
                    # Call media_info_v1 directly rather than the client.media_info()
                    # wrapper: that wrapper swallows FeedbackRequired/ChallengeRequired
                    # internally and falls through to the public scrape endpoint, which
                    # then fails with an unrelated ClientForbiddenError — masking the
                    # actual action block we need to detect and back off from.
                    full = self.client.media_info_v1(pk)
                    self._action_block_streak = 0
                except ACTION_BLOCK_EXCEPTIONS:
                    self._trip_action_block(pk, "media_info")
                    blocked = True
                except Exception:  # noqa: BLE001
                    try:
                        full = self.client.media_info(pk)
                    except Exception:  # noqa: BLE001
                        log.info("media_info failed for %s; using embedded data.", pk)

        link = _media_link(self.client, full) or _media_link(self.client, media)
        if not link:
            log.info("No link for shared item %s; skipping.", item_id)
            return None

        creator_pk = str(getattr(getattr(full, "user", None), "pk", "") or "")
        comments: list[PostComment] = []
        if pk and blocked:
            log.info("media_comments skipped for %s; IG action-block cooldown active.", pk)
        elif pk:
            try:
                raw = self.client.media_comments(pk, amount=self.comments_fetch_limit)
                for c in raw:
                    text = getattr(c, "text", None)
                    if not text:
                        continue
                    c_user = getattr(c, "user", None)
                    c_pk = str(getattr(c_user, "pk", "") or "")
                    comments.append(
                        PostComment(
                            text=text,
                            is_creator=bool(creator_pk) and c_pk == creator_pk,
                            likes=int(getattr(c, "like_count", 0) or 0),
                            author=getattr(c_user, "username", None),
                        )
                    )
            except ACTION_BLOCK_EXCEPTIONS:
                self._trip_action_block(pk, "media_comments")
            except Exception:  # noqa: BLE001
                log.info("media_comments failed for %s; continuing.", pk)
        comments = _select_comments(comments, self.comments_limit)

        return SharedPost(
            platform=PLATFORM,
            item_id=item_id,
            thread_id=thread_id,
            link=link,
            caption=getattr(full, "caption_text", None) or None,
            location=_location_str(full),
            comments=comments,
            media_url=_video_url(full) or _video_url(media),
            media_kind="video",
        )

    def enrich_from_url(self, item_id: str, thread_id: str, url: str) -> "SharedPost":
        from types import SimpleNamespace
        from .base import SharedPost
        try:
            pk = self.client.media_pk_from_url(url)
        except Exception:
            log.info("Could not resolve pk from URL %s; returning link-only post.", url)
            return SharedPost(platform=PLATFORM, item_id=item_id, thread_id=thread_id, link=url)
        stub = SimpleNamespace(pk=pk)
        post = self._build_post(item_id, thread_id, stub)
        if post is None:
            return SharedPost(platform=PLATFORM, item_id=item_id, thread_id=thread_id, link=url)
        return post
