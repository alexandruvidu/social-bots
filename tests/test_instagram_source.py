"""Tests for InstagramSource's comment-selection helpers — no network."""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from instagrapi.exceptions import FeedbackRequired

from bot.sources.base import PostComment
from bot.sources.instagram import InstagramSource, _looks_like_place_name, _select_comments


def _make_source(client):
    # Bypass __init__ (which logs into IG for real) and wire up just the
    # client + state _build_post depends on.
    src = InstagramSource.__new__(InstagramSource)
    src.client = client
    src.comments_fetch_limit = 100
    src.comments_limit = 25
    src._media_info_blocked_until = 0.0
    src._action_block_streak = 0
    return src


def test_looks_like_place_name_explicit_location_callout():
    assert _looks_like_place_name("Location: Miradouro de São Cristovão, Portugal")


def test_looks_like_place_name_capitalized_run_with_connector():
    assert _looks_like_place_name("Miradouro de São Cristovão")


def test_looks_like_place_name_rejects_plain_reaction():
    assert not _looks_like_place_name("Stunning!")
    assert not _looks_like_place_name("I promise I'll get there!")


def test_select_comments_keeps_buried_answer_over_early_reactions():
    comments = (
        [PostComment(text="Where is it?", likes=10)]
        + [PostComment(text="😍", likes=1) for _ in range(20)]
        + [PostComment(text="Location: Miradouro de São Cristovão", likes=0)]
    )
    selected = _select_comments(comments, limit=10)
    assert any("Miradouro" in c.text for c in selected)
    assert len(selected) == 10


def test_select_comments_keeps_creator_comments():
    comments = [PostComment(text="It's Lisbon", is_creator=True, likes=0)] + [
        PostComment(text="😍", likes=5) for _ in range(20)
    ]
    selected = _select_comments(comments, limit=5)
    assert any(c.is_creator for c in selected)


def test_select_comments_no_op_under_limit():
    comments = [PostComment(text="hi", likes=0) for _ in range(3)]
    assert _select_comments(comments, limit=10) == comments


def test_build_post_backs_off_after_feedback_required():
    client = MagicMock()
    client.media_info_v1.side_effect = FeedbackRequired("feedback_required")
    client.media_link.side_effect = Exception("also blocked")
    src = _make_source(client)
    media = SimpleNamespace(pk="123", code=None, video_url=None)

    first = src._build_post("item1", "thread1", media)
    assert first is None  # no code/video_url on the stub, so still no link
    assert client.media_info_v1.call_count == 1
    assert client.media_info.call_count == 0  # never falls through to the public scrape
    assert src._media_info_blocked_until > 0.0

    # A second share arriving during the cooldown shouldn't retry the blocked call.
    second = src._build_post("item2", "thread1", media)
    assert second is None
    assert client.media_info_v1.call_count == 1


def test_build_post_escalates_cooldown_on_repeated_blocks():
    from bot.sources.instagram import ACTION_BLOCK_BASE_COOLDOWN_S

    client = MagicMock()
    client.media_info_v1.side_effect = FeedbackRequired("feedback_required")
    client.media_link.side_effect = Exception("also blocked")
    src = _make_source(client)
    media = SimpleNamespace(pk="123", code=None, video_url=None)

    src._build_post("item1", "thread1", media)
    first_cooldown = src._media_info_blocked_until
    assert first_cooldown == pytest.approx(time.monotonic() + ACTION_BLOCK_BASE_COOLDOWN_S, abs=2)

    # Force the cooldown to have already expired so the next share retries
    # and hits the block again — that repeat should double the wait, not
    # repeat the same (apparently too-short) cooldown.
    src._media_info_blocked_until = 0.0
    src._build_post("item2", "thread1", media)
    second_cooldown = src._media_info_blocked_until
    assert second_cooldown == pytest.approx(time.monotonic() + ACTION_BLOCK_BASE_COOLDOWN_S * 2, abs=2)


def test_build_post_skips_comments_during_action_block_cooldown():
    client = MagicMock()
    src = _make_source(client)
    src._media_info_blocked_until = time.monotonic() + 999  # already in cooldown
    media = SimpleNamespace(pk="123", code="abc", video_url=None)

    post = src._build_post("item1", "thread1", media)

    assert post is not None
    assert post.comments == []
    client.media_info_v1.assert_not_called()
    client.media_comments.assert_not_called()


def test_build_post_trips_cooldown_on_comments_feedback_required():
    client = MagicMock()
    client.media_info_v1.return_value = SimpleNamespace(pk="123", code="abc", video_url=None)
    client.media_comments.side_effect = FeedbackRequired("feedback_required")
    src = _make_source(client)
    media = SimpleNamespace(pk="123", code="abc", video_url=None)

    first = src._build_post("item1", "thread1", media)
    assert first is not None
    assert first.comments == []
    assert src._media_info_blocked_until > 0.0
    assert client.media_comments.call_count == 1

    # A second share during the cooldown shouldn't retry the blocked comments call.
    second = src._build_post("item2", "thread1", media)
    assert second is not None
    assert second.comments == []
    assert client.media_comments.call_count == 1
    # media_info_v1 also shouldn't be retried — it's the same account-level block.
    assert client.media_info_v1.call_count == 1


def test_build_post_retries_media_info_after_ordinary_failure():
    client = MagicMock()
    client.media_info_v1.side_effect = Exception("transient network error")
    client.media_info.side_effect = Exception("public fallback also fails")
    client.media_link.side_effect = Exception("also fails")
    src = _make_source(client)
    media = SimpleNamespace(pk="123", code=None, video_url=None)

    src._build_post("item1", "thread1", media)
    src._build_post("item2", "thread1", media)

    assert client.media_info_v1.call_count == 2
    assert src._media_info_blocked_until == 0.0
