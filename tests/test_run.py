"""run.py orchestration tests — all offline, no network."""
from unittest.mock import MagicMock, patch

import pytest

from bot.run import drain_retry_queue, handle_posts, handle_replies
from bot.sources.base import PostComment, SharedPost, TextReply


def _make_post(media_url=None, media_kind=None, comments=None):
    return SharedPost(
        platform="instagram",
        item_id="msg1",
        thread_id="t1",
        link="https://www.instagram.com/p/ABC/",
        caption="beautiful sunset",
        media_url=media_url,
        media_kind=media_kind,
        comments=comments or [],
    )


def _make_cfg(threshold=0.4):
    cfg = MagicMock()
    cfg.confidence_threshold = threshold
    cfg.model = "gemini-2.5-flash"
    cfg.trek_url = None
    cfg.trek_api_token = None
    return cfg


def _make_store():
    store = MagicMock()
    store.is_processed.return_value = False
    store.get_destinations_for_link.return_value = []
    return store


def test_video_fallback_called_when_text_extract_fails():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(media_url="https://cdn.instagram.com/video.mp4")

    low_confidence = Extracted(destination=None, confidence=0.0)
    high_confidence = Extracted(destination="Bali, Indonesia", confidence=0.9, source_field="video")

    with patch("bot.run.extract", return_value=low_confidence), \
         patch("bot.run.analyze_media", return_value=high_confidence) as mock_media:
        handle_posts(store, source, [post], cfg)

    mock_media.assert_called_once_with(
        "https://cdn.instagram.com/video.mp4", None, model=cfg.model
    )
    store.save_destination.assert_called_once()
    saved_kwargs = store.save_destination.call_args.kwargs
    assert saved_kwargs["destination"] == "Bali, Indonesia"


def test_video_fallback_not_called_when_text_succeeds():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(media_url="https://cdn.instagram.com/video.mp4")

    success = Extracted(destination="Paris, France", confidence=0.95, source_field="caption")

    with patch("bot.run.extract", return_value=success), \
         patch("bot.run.analyze_media") as mock_media:
        handle_posts(store, source, [post], cfg)

    mock_media.assert_not_called()


def test_video_fallback_not_called_when_no_media_url():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(media_url=None)

    low_confidence = Extracted(destination=None, confidence=0.0)

    with patch("bot.run.extract", return_value=low_confidence), \
         patch("bot.run.analyze_media") as mock_media:
        handle_posts(store, source, [post], cfg)

    mock_media.assert_not_called()
    source.reply.assert_called_once()


def test_extract_receives_all_available_text_signals_in_one_call():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(comments=[PostComment(text="It's Lisbon!", likes=5)])

    success = Extracted(destination="Paris, France", confidence=0.95, source_field="caption")

    with patch("bot.run.extract", return_value=success) as mock_extract, \
         patch("bot.run.analyze_media") as mock_media:
        handle_posts(store, source, [post], cfg)

    mock_extract.assert_called_once_with(post.caption, post.location, post.comments, model=cfg.model)
    mock_media.assert_not_called()


def test_video_is_tried_after_all_text_signals_fail():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(
        media_url="https://cdn.instagram.com/video.mp4",
        comments=[PostComment(text="It's Lisbon!", likes=5)],
    )

    low_confidence = Extracted(destination=None, confidence=0.0)
    video_success = Extracted(destination="Bali, Indonesia", confidence=0.9, source_field="video")

    with patch("bot.run.extract", return_value=low_confidence) as mock_extract, \
         patch("bot.run.analyze_media", return_value=video_success) as mock_media:
        handle_posts(store, source, [post], cfg)

    mock_media.assert_called_once_with("https://cdn.instagram.com/video.mp4", None, model=cfg.model)
    # Video succeeded, and there was only one text extraction call.
    assert mock_extract.call_count == 1
    saved_kwargs = store.save_destination.call_args.kwargs
    assert saved_kwargs["destination"] == "Bali, Indonesia"


def test_comments_are_not_sent_in_a_second_gemini_call():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    comments = [PostComment(text="It's Lisbon!", is_creator=True, likes=5)]
    post = _make_post(media_url="https://cdn.instagram.com/video.mp4", comments=comments)

    low_confidence = Extracted(destination=None, confidence=0.0)
    with patch("bot.run.extract", return_value=low_confidence) as mock_extract, \
         patch("bot.run.analyze_media", return_value=low_confidence) as mock_media:
        handle_posts(store, source, [post], cfg)

    mock_media.assert_called_once()
    mock_extract.assert_called_once_with(post.caption, post.location, comments, model=cfg.model)
    source.reply.assert_called_once()


def test_no_comments_retry_when_post_has_no_comments():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(media_url="https://cdn.instagram.com/video.mp4")  # no comments

    low_confidence = Extracted(destination=None, confidence=0.0)

    with patch("bot.run.extract", return_value=low_confidence) as mock_extract, \
         patch("bot.run.analyze_media", return_value=low_confidence):
        handle_posts(store, source, [post], cfg)

    assert mock_extract.call_count == 1
    source.reply.assert_called_once()


def test_multiple_places_all_saved_and_reported():
    from bot.extract import Extracted, Place

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post()

    result = Extracted(
        destination="Tokyo, Japan",
        confidence=0.9,
        source_field="caption",
        more_places=[
            Place(destination="Kyoto, Japan", confidence=0.8, source_field="caption"),
            Place(destination="dubious guess", confidence=0.1, source_field="comments"),
        ],
    )

    with patch("bot.run.extract", return_value=result):
        handle_posts(store, source, [post], cfg)

    # Primary + the one place above threshold; the low-confidence one is dropped.
    assert store.save_destination.call_count == 2
    saved_destinations = {
        call.kwargs["destination"] for call in store.save_destination.call_args_list
    }
    assert saved_destinations == {"Tokyo, Japan", "Kyoto, Japan"}

    source.reply.assert_called_once()
    reply_text = source.reply.call_args.args[1]
    assert "Tokyo, Japan" in reply_text
    assert "Kyoto, Japan" in reply_text
    assert "dubious guess" not in reply_text


def test_already_saved_destination_is_reported():
    from bot.extract import Extracted

    store = _make_store()
    store.save_destination.return_value = False  # already in the database
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post()

    result = Extracted(destination="Tokyo, Japan", confidence=0.9, source_field="caption")

    with patch("bot.run.extract", return_value=result):
        handle_posts(store, source, [post], cfg)

    source.reply.assert_called_once()
    reply_text = source.reply.call_args.args[1]
    assert "Tokyo, Japan" in reply_text
    assert "Already saved" in reply_text


def test_mixed_new_and_already_saved_destinations_reported_together():
    from bot.extract import Extracted, Place

    store = _make_store()
    store.save_destination.side_effect = [True, False]  # first new, second duplicate
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post()

    result = Extracted(
        destination="Tokyo, Japan",
        confidence=0.9,
        source_field="caption",
        more_places=[Place(destination="Kyoto, Japan", confidence=0.8, source_field="caption")],
    )

    with patch("bot.run.extract", return_value=result):
        handle_posts(store, source, [post], cfg)

    source.reply.assert_called_once()
    reply_text = source.reply.call_args.args[1]
    assert "Saved: Tokyo, Japan" in reply_text
    assert "Already saved: Kyoto, Japan" in reply_text


def test_link_already_in_db_skips_gemini_and_reports_existing():
    store = _make_store()
    store.get_destinations_for_link.return_value = [
        {"destination": "Tokyo, Japan", "landmark": "Shibuya Crossing", "place_type": "landmark"},
        {"destination": "Tokyo, Japan", "landmark": None, "place_type": None},
    ]
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post()

    with patch("bot.run.extract") as mock_extract, \
         patch("bot.run.analyze_media") as mock_media:
        handle_posts(store, source, [post], cfg)

    mock_extract.assert_not_called()
    mock_media.assert_not_called()
    store.save_destination.assert_not_called()
    store.mark_processed.assert_called_once_with(post.platform, post.item_id)

    source.reply.assert_called_once()
    reply_text = source.reply.call_args.args[1]
    assert "Already saved" in reply_text
    assert "Shibuya Crossing" in reply_text


def test_video_fallback_asks_user_when_video_also_fails():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(media_url="https://cdn.instagram.com/video.mp4")

    low = Extracted(destination=None, confidence=0.0)

    with patch("bot.run.extract", return_value=low), \
         patch("bot.run.analyze_media", return_value=low):
        handle_posts(store, source, [post], cfg)

    source.reply.assert_called_once()
    store.save_destination.assert_not_called()


def test_rate_limited_post_is_queued_not_dropped():
    from bot.extract import RateLimitedError

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post()

    with patch("bot.run.extract", side_effect=RateLimitedError(30.0)):
        handle_posts(store, source, [post], cfg)

    store.enqueue_retry.assert_called_once()
    args = store.enqueue_retry.call_args.args
    assert args[0] == post.platform
    assert args[1] == post.item_id
    assert args[3] == 30.0
    store.mark_processed.assert_not_called()
    source.reply.assert_not_called()


def test_drain_retry_queue_processes_due_post_and_deletes_row():
    from bot.extract import Extracted
    from bot.run import _serialize_post

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post()
    row = {"id": 7, "payload": _serialize_post(post), "attempts": 0}
    store.due_retries.return_value = [row]

    success = Extracted(destination="Lisbon, Portugal", confidence=0.9, source_field="caption")
    with patch("bot.run.extract", return_value=success):
        drain_retry_queue(store, source, cfg)

    store.delete_retry.assert_called_once_with(7)
    store.reschedule_retry.assert_not_called()


def test_drain_retry_queue_reschedules_if_still_rate_limited():
    from bot.extract import RateLimitedError
    from bot.run import _serialize_post

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post()
    row = {"id": 7, "payload": _serialize_post(post), "attempts": 1}
    store.due_retries.return_value = [row]

    with patch("bot.run.extract", side_effect=RateLimitedError(45.0)):
        drain_retry_queue(store, source, cfg)

    store.reschedule_retry.assert_called_once_with(7, 45.0)
    store.delete_retry.assert_not_called()


def test_drain_retry_queue_gives_up_after_max_attempts():
    from bot.extract import RateLimitedError
    from bot.run import _MAX_RETRY_ATTEMPTS, _serialize_post

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post()
    row = {"id": 7, "payload": _serialize_post(post), "attempts": _MAX_RETRY_ATTEMPTS - 1}
    store.due_retries.return_value = [row]

    with patch("bot.run.extract", side_effect=RateLimitedError(45.0)):
        drain_retry_queue(store, source, cfg)

    store.delete_retry.assert_called_once_with(7)
    store.reschedule_retry.assert_not_called()


def test_deserialize_post_migrates_legacy_video_url():
    """A retry row queued before the media_url rename must still load."""
    import json
    from bot.run import _deserialize_post

    legacy = json.dumps({
        "platform": "instagram",
        "item_id": "msg1",
        "thread_id": "t1",
        "link": "https://www.instagram.com/p/ABC/",
        "caption": "sunset",
        "location": None,
        "comments": [],
        "video_url": "https://cdn.example.com/v.mp4",
    })

    post = _deserialize_post(legacy)

    assert post.media_url == "https://cdn.example.com/v.mp4"
    assert post.media_kind == "video"


def test_deserialize_post_drops_unknown_keys():
    """Unrecognised keys from any future schema drift must not raise."""
    import json
    from bot.run import _deserialize_post

    payload = json.dumps({
        "platform": "instagram",
        "item_id": "msg1",
        "thread_id": "t1",
        "link": "https://www.instagram.com/p/ABC/",
        "some_removed_field": "whatever",
    })

    post = _deserialize_post(payload)

    assert post.item_id == "msg1"
    assert post.media_url is None


# ── I4: the cron path must not silently log in to the private API ──────────

def test_run_once_refuses_without_opt_in(monkeypatch):
    """bot.run drives instagrapi; reaching it by accident re-breaks the account."""
    from bot.run import run_once

    monkeypatch.delenv("ALLOW_PRIVATE_API", raising=False)

    with patch("bot.sources.instagram.InstagramSource") as mock_src, \
         patch("bot.config.Config.load") as mock_load:
        with pytest.raises(SystemExit) as exc:
            run_once()

    mock_src.assert_not_called()
    mock_load.assert_not_called()
    message = str(exc.value)
    assert "private" in message.lower()
    assert "ALLOW_PRIVATE_API=1" in message
    assert "bot.webhook" in message


def test_run_once_refuses_when_credentials_are_missing(monkeypatch):
    """ig_username/ig_password are optional now — say so instead of failing opaquely."""
    from bot.run import run_once

    monkeypatch.setenv("ALLOW_PRIVATE_API", "1")
    cfg = _make_cfg()
    cfg.ig_username = None
    cfg.ig_password = None

    with patch("bot.sources.instagram.InstagramSource") as mock_src, \
         patch("bot.config.Config.load", return_value=cfg):
        with pytest.raises(SystemExit) as exc:
            run_once()

    mock_src.assert_not_called()
    assert "IG_USERNAME" in str(exc.value)


def test_instagram_source_rejects_missing_credentials():
    """The typed contract said `str`, but Config now hands it `str | None`."""
    from pathlib import Path
    from bot.sources.instagram import InstagramSource

    with patch("bot.sources.instagram.login_client") as login:
        with pytest.raises(ValueError, match="IG_USERNAME"):
            InstagramSource(username=None, password=None, allowed_sender_id="111",
                            session_path=Path("/nonexistent/session.json"))

    login.assert_not_called()


def _make_pending(link="https://www.instagram.com/p/ABC/", ask_msg_id="ask_1"):
    return {"link": link, "caption_snippet": "snip", "ask_msg_id": ask_msg_id}


def test_extract_skipped_when_no_caption_or_location():
    """No text means no prompt worth sending — go straight to the media."""
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(media_url="https://lookaside.fbsbx.com/x", media_kind="video")
    post.caption = None
    post.location = None

    found = Extracted(destination="Bali, Indonesia", confidence=0.9, source_field="video")

    with patch("bot.run.extract") as mock_extract, \
         patch("bot.run.analyze_media", return_value=found) as mock_media:
        handle_posts(store, source, [post], cfg)

    mock_extract.assert_not_called()
    mock_media.assert_called_once_with(
        "https://lookaside.fbsbx.com/x", "video", model=cfg.model
    )
    assert store.save_destination.call_args.kwargs["destination"] == "Bali, Indonesia"


def test_media_reply_resolves_pending_row():
    from bot.extract import Extracted
    from bot.run import handle_media_replies
    from bot.sources.base import MediaReply

    store = _make_store()
    store.get_pending_by_ask_msg.return_value = _make_pending()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    reply = MediaReply(platform="instagram", item_id="m2", thread_id="t1",
                       media_url="https://lookaside.fbsbx.com/shot",
                       reply_to_item_id="ask_1")

    found = Extracted(destination="Lofoten, Norway", confidence=0.9,
                      source_field="screenshot")

    with patch("bot.run.analyze_image", return_value=found):
        handle_media_replies(store, source, [reply], cfg)

    kwargs = store.save_destination.call_args.kwargs
    assert kwargs["destination"] == "Lofoten, Norway"
    assert kwargs["link"] == "https://www.instagram.com/p/ABC/"
    assert kwargs["source_field"] == "screenshot"
    store.clear_pending_by_ask_msg.assert_called_once_with("instagram", "ask_1")
    store.mark_processed.assert_called_once_with("instagram", "m2")


def test_media_reply_low_confidence_leaves_pending_open():
    """A failed screenshot must not close the row — the text reply still works."""
    from bot.extract import Extracted
    from bot.run import MEDIA_RETRY_TEXT, handle_media_replies
    from bot.sources.base import MediaReply

    store = _make_store()
    store.get_pending_by_ask_msg.return_value = _make_pending()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    reply = MediaReply(platform="instagram", item_id="m2", thread_id="t1",
                       media_url="https://lookaside.fbsbx.com/shot",
                       reply_to_item_id="ask_1")

    with patch("bot.run.analyze_image", return_value=Extracted(confidence=0.0)):
        handle_media_replies(store, source, [reply], cfg)

    store.save_destination.assert_not_called()
    store.clear_pending_by_ask_msg.assert_not_called()
    store.clear_pending.assert_not_called()
    assert source.reply.call_args.args[1] == MEDIA_RETRY_TEXT


def test_media_reply_rate_limited_leaves_row_unprocessed():
    """The open pending row IS the retry mechanism — don't consume the message."""
    from bot.extract import RateLimitedError
    from bot.run import MEDIA_RATE_LIMIT_TEXT, handle_media_replies
    from bot.sources.base import MediaReply

    store = _make_store()
    store.get_pending_by_ask_msg.return_value = _make_pending()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    reply = MediaReply(platform="instagram", item_id="m2", thread_id="t1",
                       media_url="https://lookaside.fbsbx.com/shot",
                       reply_to_item_id="ask_1")

    with patch("bot.run.analyze_image", side_effect=RateLimitedError(300.0)):
        handle_media_replies(store, source, [reply], cfg)

    store.save_destination.assert_not_called()
    store.mark_processed.assert_not_called()
    assert source.reply.call_args.args[1] == MEDIA_RATE_LIMIT_TEXT


def test_media_reply_without_pending_is_ignored():
    from bot.run import handle_media_replies
    from bot.sources.base import MediaReply

    store = _make_store()
    store.get_pending_by_ask_msg.return_value = None
    store.get_pending.return_value = None
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    reply = MediaReply(platform="instagram", item_id="m2", thread_id="t1",
                       media_url="https://lookaside.fbsbx.com/shot",
                       reply_to_item_id=None)

    with patch("bot.run.analyze_image") as mock_img:
        handle_media_replies(store, source, [reply], cfg)

    mock_img.assert_not_called()
    store.save_destination.assert_not_called()
    store.mark_processed.assert_called_once_with("instagram", "m2")


def _make_text_reply(text="Kyoto, Japan", reply_to_item_id="ask_1"):
    return TextReply(
        platform="instagram",
        item_id="reply1",
        thread_id="t1",
        text=text,
        reply_to_item_id=reply_to_item_id,
    )


def test_trek_push_called_when_configured_and_no_failure_note_on_success():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    cfg.trek_url = "http://trek.local:3000"
    cfg.trek_api_token = "trek_abc"
    post = _make_post()

    result = Extracted(
        destination="Tokyo, Japan", confidence=0.9, source_field="caption", topic="best ramen spots",
    )

    with patch("bot.run.extract", return_value=result), \
         patch("bot.run.push_destination", return_value=True) as mock_push:
        handle_posts(store, source, [post], cfg)

    mock_push.assert_called_once_with(
        cfg,
        platform="instagram",
        link=post.link,
        destination="Tokyo, Japan",
        landmark=None,
        place_type=None,
        topic="best ramen spots",
        caption_snippet="beautiful sunset",
    )
    reply_text = source.reply.call_args.args[1]
    assert "Couldn't sync to TREK" not in reply_text


def test_trek_push_failure_appends_note_to_reply():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    cfg.trek_url = "http://trek.local:3000"
    cfg.trek_api_token = "trek_abc"
    post = _make_post()

    result = Extracted(destination="Tokyo, Japan", confidence=0.9, source_field="caption")

    with patch("bot.run.extract", return_value=result), \
         patch("bot.run.push_destination", return_value=False):
        handle_posts(store, source, [post], cfg)

    reply_text = source.reply.call_args.args[1]
    assert "Couldn't sync to TREK" in reply_text


def test_trek_push_skipped_when_not_configured():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()  # trek_url/trek_api_token default to None
    post = _make_post()

    result = Extracted(destination="Tokyo, Japan", confidence=0.9, source_field="caption")

    with patch("bot.run.extract", return_value=result), \
         patch("bot.run.push_destination") as mock_push:
        handle_posts(store, source, [post], cfg)

    mock_push.assert_not_called()


def test_handle_replies_pushes_to_trek_silently_on_success():
    store = _make_store()
    store.get_pending_by_ask_msg.return_value = {
        "link": "https://www.instagram.com/p/ABC/", "caption_snippet": None, "ask_msg_id": "ask_1",
    }
    store.save_destination.return_value = True
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    cfg.trek_url = "http://trek.local:3000"
    cfg.trek_api_token = "trek_abc"
    reply = _make_text_reply()

    with patch("bot.run.push_destination", return_value=True) as mock_push:
        handle_replies(store, source, [reply], cfg)

    mock_push.assert_called_once_with(
        cfg,
        platform="instagram",
        link="https://www.instagram.com/p/ABC/",
        destination="Kyoto, Japan",
        landmark=None,
        place_type=None,
        topic=None,
        caption_snippet=None,
    )
    source.reply.assert_not_called()  # stays silent on success, matches existing behavior


def test_handle_replies_replies_with_failure_note_when_push_fails():
    store = _make_store()
    store.get_pending_by_ask_msg.return_value = {
        "link": "https://www.instagram.com/p/ABC/", "caption_snippet": None, "ask_msg_id": "ask_1",
    }
    store.save_destination.return_value = True
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    cfg.trek_url = "http://trek.local:3000"
    cfg.trek_api_token = "trek_abc"
    reply = _make_text_reply()

    with patch("bot.run.push_destination", return_value=False):
        handle_replies(store, source, [reply], cfg)

    source.reply.assert_called_once()
    assert "Couldn't sync to TREK" in source.reply.call_args.args[1]


def test_handle_replies_skips_trek_when_not_configured():
    store = _make_store()
    store.get_pending_by_ask_msg.return_value = {
        "link": "https://www.instagram.com/p/ABC/", "caption_snippet": None, "ask_msg_id": "ask_1",
    }
    store.save_destination.return_value = True
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    reply = _make_text_reply()

    with patch("bot.run.push_destination") as mock_push:
        handle_replies(store, source, [reply], cfg)

    mock_push.assert_not_called()
    source.reply.assert_not_called()
