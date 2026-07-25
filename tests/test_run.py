"""run.py orchestration tests — all offline, no network."""
from unittest.mock import MagicMock, patch

import pytest

from bot.run import drain_retry_queue, handle_posts
from bot.sources.base import PostComment, SharedPost


def _make_post(video_url=None, comments=None):
    return SharedPost(
        platform="instagram",
        item_id="msg1",
        thread_id="t1",
        link="https://www.instagram.com/p/ABC/",
        caption="beautiful sunset",
        video_url=video_url,
        comments=comments or [],
    )


def _make_cfg(threshold=0.4):
    cfg = MagicMock()
    cfg.confidence_threshold = threshold
    cfg.model = "gemini-2.5-flash"
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
    post = _make_post(video_url="https://cdn.instagram.com/video.mp4")

    low_confidence = Extracted(destination=None, confidence=0.0)
    high_confidence = Extracted(destination="Bali, Indonesia", confidence=0.9, source_field="video")

    with patch("bot.run.extract", return_value=low_confidence), \
         patch("bot.run.analyze_video", return_value=high_confidence) as mock_video:
        handle_posts(store, source, [post], cfg)

    mock_video.assert_called_once_with(
        "https://cdn.instagram.com/video.mp4", model=cfg.model
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
    post = _make_post(video_url="https://cdn.instagram.com/video.mp4")

    success = Extracted(destination="Paris, France", confidence=0.95, source_field="caption")

    with patch("bot.run.extract", return_value=success), \
         patch("bot.run.analyze_video") as mock_video:
        handle_posts(store, source, [post], cfg)

    mock_video.assert_not_called()


def test_video_fallback_not_called_when_no_video_url():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(video_url=None)

    low_confidence = Extracted(destination=None, confidence=0.0)

    with patch("bot.run.extract", return_value=low_confidence), \
         patch("bot.run.analyze_video") as mock_video:
        handle_posts(store, source, [post], cfg)

    mock_video.assert_not_called()
    source.reply.assert_called_once()


def test_extract_called_without_comments_first():
    """Comments are an unreliable, risk-laden signal — caption/location alone
    is tried first so a clean post never needs them at all."""
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(comments=[PostComment(text="It's Lisbon!", likes=5)])

    success = Extracted(destination="Paris, France", confidence=0.95, source_field="caption")

    with patch("bot.run.extract", return_value=success) as mock_extract, \
         patch("bot.run.analyze_video") as mock_video:
        handle_posts(store, source, [post], cfg)

    mock_extract.assert_called_once_with(post.caption, post.location, [], model=cfg.model)
    mock_video.assert_not_called()


def test_video_tried_before_comments_when_caption_extraction_fails():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(
        video_url="https://cdn.instagram.com/video.mp4",
        comments=[PostComment(text="It's Lisbon!", likes=5)],
    )

    low_confidence = Extracted(destination=None, confidence=0.0)
    video_success = Extracted(destination="Bali, Indonesia", confidence=0.9, source_field="video")

    with patch("bot.run.extract", return_value=low_confidence) as mock_extract, \
         patch("bot.run.analyze_video", return_value=video_success) as mock_video:
        handle_posts(store, source, [post], cfg)

    mock_video.assert_called_once_with("https://cdn.instagram.com/video.mp4", model=cfg.model)
    # Video succeeded, so the comments-augmented retry never had to run.
    assert mock_extract.call_count == 1
    saved_kwargs = store.save_destination.call_args.kwargs
    assert saved_kwargs["destination"] == "Bali, Indonesia"


def test_falls_back_to_comments_when_caption_and_video_both_fail():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    comments = [PostComment(text="It's Lisbon!", is_creator=True, likes=5)]
    post = _make_post(video_url="https://cdn.instagram.com/video.mp4", comments=comments)

    low_confidence = Extracted(destination=None, confidence=0.0)
    comments_success = Extracted(destination="Lisbon, Portugal", confidence=0.9, source_field="comments")

    def fake_extract(caption, location, post_comments, model=None):
        if post_comments:
            return comments_success
        return low_confidence

    with patch("bot.run.extract", side_effect=fake_extract) as mock_extract, \
         patch("bot.run.analyze_video", return_value=low_confidence) as mock_video:
        handle_posts(store, source, [post], cfg)

    mock_video.assert_called_once()
    assert mock_extract.call_count == 2
    first_call, second_call = mock_extract.call_args_list
    assert first_call.args[2] == []
    assert second_call.args[2] == comments
    saved_kwargs = store.save_destination.call_args.kwargs
    assert saved_kwargs["destination"] == "Lisbon, Portugal"


def test_no_comments_retry_when_post_has_no_comments():
    from bot.extract import Extracted

    store = _make_store()
    source = MagicMock()
    source.platform = "instagram"
    cfg = _make_cfg()
    post = _make_post(video_url="https://cdn.instagram.com/video.mp4")  # no comments

    low_confidence = Extracted(destination=None, confidence=0.0)

    with patch("bot.run.extract", return_value=low_confidence) as mock_extract, \
         patch("bot.run.analyze_video", return_value=low_confidence):
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
         patch("bot.run.analyze_video") as mock_video:
        handle_posts(store, source, [post], cfg)

    mock_extract.assert_not_called()
    mock_video.assert_not_called()
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
    post = _make_post(video_url="https://cdn.instagram.com/video.mp4")

    low = Extracted(destination=None, confidence=0.0)

    with patch("bot.run.extract", return_value=low), \
         patch("bot.run.analyze_video", return_value=low):
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
