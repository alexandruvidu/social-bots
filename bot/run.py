"""Batch orchestrator: one pass over new DMs, then exit. Driven by cron.

    python -m bot.run
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, fields
from types import SimpleNamespace

from .config import Config
from .extract import RateLimitedError, analyze_video, extract, is_success
from .sources.base import PostComment, SharedPost, Source, TextReply
from .sources.instagram import InstagramSource

log = logging.getLogger("bot")

# Give up on a rate-limited post after this many retry attempts (each spaced
# by the wait Gemini itself suggested), rather than queuing it forever.
_MAX_RETRY_ATTEMPTS = 8

ASK_TEXT = (
    "Couldn't find a destination for this one — reply with the place "
    "and I'll save it."
)


def _snippet(caption: str | None, n: int = 200) -> str | None:
    if not caption:
        return None
    return caption[:n]


def _format_place(place) -> str:
    text = place.destination
    if place.landmark:
        text += f" — {place.landmark}"
        if place.place_type:
            text += f" ({place.place_type})"
    return text


def _format_saved_text(places: list, link: str) -> str:
    if len(places) == 1:
        return f"Saved: {_format_place(places[0])}\n{link}"
    lines = [f"Saved {len(places)} places:"]
    lines.extend(f"- {_format_place(p)}" for p in places)
    lines.append(link)
    return "\n".join(lines)


def _format_result_text(saved_places: list, existing_places: list, link: str) -> str:
    """Reply text covering newly saved places and ones already in the database."""
    if not existing_places:
        return _format_saved_text(saved_places, link)
    lines = []
    if saved_places:
        if len(saved_places) == 1:
            lines.append(f"Saved: {_format_place(saved_places[0])}")
        else:
            lines.append(f"Saved {len(saved_places)} places:")
            lines.extend(f"- {_format_place(p)}" for p in saved_places)
    if len(existing_places) == 1:
        lines.append(f"Already saved: {_format_place(existing_places[0])}")
    else:
        lines.append(f"Already saved {len(existing_places)} of these places:")
        lines.extend(f"- {_format_place(p)}" for p in existing_places)
    lines.append(link)
    return "\n".join(lines)


def handle_replies(store, source: Source, replies: list[TextReply]) -> None:
    for reply in replies:
        if store.is_processed(reply.platform, reply.item_id):
            continue
        # Prefer matching by the specific ask message the user replied to; fall
        # back to a thread-level match for pending rows without an ask_msg_id.
        pending = None
        if reply.reply_to_item_id:
            pending = store.get_pending_by_ask_msg(reply.platform, reply.reply_to_item_id)
        if pending is None:
            pending = store.get_pending(reply.platform, reply.thread_id)
        if pending is not None:
            saved = store.save_destination(
                platform=reply.platform,
                link=pending["link"],
                destination=reply.text,
                confidence=1.0,
                source_field="user",
                caption_snippet=pending["caption_snippet"],
                sender=source.platform,
            )
            ask_msg_id = pending["ask_msg_id"]
            if ask_msg_id:
                store.clear_pending_by_ask_msg(reply.platform, ask_msg_id)
            else:
                store.clear_pending(reply.platform, reply.thread_id)
            log.info(
                "Saved user-provided destination %r for %s (new=%s)",
                reply.text,
                pending["link"],
                saved,
            )
        # Mark processed whether or not it resolved a pending row, so a chatty
        # thread doesn't get re-scanned forever.
        store.mark_processed(reply.platform, reply.item_id)


def _serialize_post(post: SharedPost) -> str:
    data = asdict(post)
    return json.dumps(data)


_SHARED_POST_FIELDS = {f.name for f in fields(SharedPost)}


def _deserialize_post(payload: str) -> SharedPost:
    data = json.loads(payload)
    # A queued retry row can outlive a restart, so it may predate the
    # video_url -> media_url rename. Migrate the old key, then drop anything
    # else we no longer recognise rather than blowing up on it.
    if "video_url" in data and "media_url" not in data:
        legacy = data.pop("video_url")
        data["media_url"] = legacy
        data.setdefault("media_kind", "video" if legacy else None)
    data = {k: v for k, v in data.items() if k in _SHARED_POST_FIELDS}
    data["comments"] = [PostComment(**c) for c in data.get("comments", [])]
    return SharedPost(**data)


def _process_post(store, source: Source, post: SharedPost, cfg: Config) -> None:
    """Process a single post: extract, save, reply, mark processed.

    Raises RateLimitedError if Gemini is quota-limited, so callers can queue
    the post for a background retry instead of dropping it.
    """
    existing_rows = store.get_destinations_for_link(post.platform, post.link)
    if existing_rows:
        existing_places = [
            SimpleNamespace(
                destination=row["destination"],
                landmark=row["landmark"],
                place_type=row["place_type"],
            )
            for row in existing_rows
        ]
        log.info("Link already in db; skipping Gemini for %s", post.link)
        source.reply(
            post.thread_id,
            _format_result_text([], existing_places, post.link),
            reply_to_item_id=post.item_id,
        )
        store.mark_processed(post.platform, post.item_id)
        return
    try:
        # Comments are the least trustworthy signal (unreliable answers, and
        # the only one that needs the risky/action-block-prone private-API
        # call) — try caption/location alone, then the video itself, before
        # ever spending a comments-augmented call on it.
        result = extract(post.caption, post.location, [], model=cfg.model)
        if not is_success(result, cfg.confidence_threshold) and post.media_url:
            log.info("Caption/location extraction failed; trying video analysis for %s", post.link)
            try:
                result = analyze_video(post.media_url, model=cfg.model)
            except RateLimitedError:
                raise
            except Exception:
                log.warning("Video analysis failed for %s; falling back to comments.", post.link, exc_info=True)
        if not is_success(result, cfg.confidence_threshold) and post.comments:
            log.info("Video analysis failed; trying comments-augmented extraction for %s", post.link)
            result = extract(post.caption, post.location, post.comments, model=cfg.model)
        if is_success(result, cfg.confidence_threshold):
            places = [result] + [
                p for p in result.more_places
                if is_success(p, cfg.confidence_threshold)
            ]
            saved_places = []
            existing_places = []
            for place in places:
                saved = store.save_destination(
                    platform=post.platform,
                    link=post.link,
                    destination=place.destination,
                    landmark=place.landmark,
                    place_type=place.place_type,
                    confidence=place.confidence,
                    source_field=place.source_field,
                    caption_snippet=_snippet(post.caption),
                    sender=source.platform,
                )
                log.info(
                    "Saved %r (landmark=%r, place_type=%r, %.2f, %s) for %s (new=%s)",
                    place.destination,
                    place.landmark,
                    place.place_type,
                    place.confidence,
                    place.source_field,
                    post.link,
                    saved,
                )
                if saved:
                    saved_places.append(place)
                else:
                    existing_places.append(place)
            if saved_places or existing_places:
                source.reply(
                    post.thread_id,
                    _format_result_text(saved_places, existing_places, post.link),
                    reply_to_item_id=post.item_id,
                )
        else:
            ask_msg_id = source.reply(
                post.thread_id, ASK_TEXT, reply_to_item_id=post.item_id
            )
            store.add_pending(
                platform=post.platform,
                thread_id=post.thread_id,
                link=post.link,
                caption_snippet=_snippet(post.caption),
                ask_msg_id=ask_msg_id,
            )
            log.info("No destination for %s; asked user.", post.link)
    except RateLimitedError:
        raise
    except Exception:  # noqa: BLE001
        log.exception("Failed processing post %s; skipping.", post.item_id)
        return
    store.mark_processed(post.platform, post.item_id)


def handle_posts(
    store, source: Source, posts: list[SharedPost], cfg: Config
) -> None:
    for post in posts:
        if store.is_processed(post.platform, post.item_id):
            continue
        try:
            _process_post(store, source, post, cfg)
        except RateLimitedError as e:
            log.warning(
                "Gemini rate limited processing %s; queuing retry in %.0fs",
                post.link, e.retry_after,
            )
            store.enqueue_retry(
                post.platform, post.item_id, _serialize_post(post), e.retry_after
            )


def drain_retry_queue(store, source: Source, cfg: Config) -> None:
    for row in store.due_retries():
        post = _deserialize_post(row["payload"])
        if store.is_processed(post.platform, post.item_id):
            store.delete_retry(row["id"])
            continue
        try:
            _process_post(store, source, post, cfg)
        except RateLimitedError as e:
            attempts = row["attempts"] + 1
            if attempts >= _MAX_RETRY_ATTEMPTS:
                log.error(
                    "Giving up on %s after %d rate-limit retries",
                    post.link, attempts,
                )
                store.delete_retry(row["id"])
            else:
                log.warning(
                    "Still rate limited processing %s; retrying in %.0fs (attempt %d/%d)",
                    post.link, e.retry_after, attempts, _MAX_RETRY_ATTEMPTS,
                )
                store.reschedule_retry(row["id"], e.retry_after)
            continue
        store.delete_retry(row["id"])


def run_once() -> None:
    from .store import Store  # local import keeps config errors first

    cfg = Config.load()
    store = Store(cfg.db_path)
    source = InstagramSource(
        username=cfg.ig_username,
        password=cfg.ig_password,
        allowed_sender_id=cfg.allowed_sender_id,
        session_path=cfg.session_path,
        comments_limit=cfg.comments_limit,
        comments_fetch_limit=cfg.comments_fetch_limit,
    )
    try:
        posts, replies = source.fetch_new()
        log.info("Fetched %d shared posts, %d replies.", len(posts), len(replies))
        # Replies first: a reply may resolve a pending row created last run.
        handle_replies(store, source, replies)
        handle_posts(store, source, posts, cfg)
        drain_retry_queue(store, source, cfg)
    finally:
        source.persist()
        store.close()


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("bot").setLevel(logging.INFO)
    try:
        run_once()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        log.exception("Batch run failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
