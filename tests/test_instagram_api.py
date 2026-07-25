"""InstagramAPISource unit tests — no network."""
from unittest.mock import MagicMock, patch
import pytest

from bot.sources.base import SharedPost, TextReply


def _make_source():
    from bot.sources.instagram_api import InstagramAPISource
    return InstagramAPISource(
        access_token="tok",
        ig_user_id="bot999",
        allowed_sender_id="sender111",
    )


def test_fetch_new_returns_empty():
    src = _make_source()
    posts, replies = src.fetch_new()
    assert posts == []
    assert replies == []


def test_reply_sends_graph_api_request():
    import httpx
    src = _make_source()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message_id": "mid_sent_1"}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = src.reply("sender111", "Hello!")

    assert result == "mid_sent_1"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "bot999" in call_kwargs.args[0]
    assert call_kwargs.kwargs["json"]["recipient"]["id"] == "sender111"
    assert call_kwargs.kwargs["json"]["message"]["text"] == "Hello!"


def test_reply_returns_none_on_failure():
    src = _make_source()
    with patch("httpx.post", side_effect=Exception("network error")):
        result = src.reply("sender111", "Hello!")
    assert result is None


def test_build_post_from_event_ig_reel():
    src = _make_source()
    messaging = {
        "sender": {"id": "sender111"},
        "message": {
            "mid": "msg_abc",
            "attachments": [{
                "type": "ig_reel",
                "payload": {"url": "https://www.instagram.com/reel/XYZ/"},
            }],
        },
    }
    post = src.build_post_from_event(messaging)
    assert isinstance(post, SharedPost)
    assert post.item_id == "msg_abc"
    assert post.thread_id == "sender111"
    assert post.link == "https://www.instagram.com/reel/XYZ/"


def test_build_post_from_event_unknown_type_returns_none():
    src = _make_source()
    messaging = {
        "sender": {"id": "sender111"},
        "message": {
            "mid": "msg_abc",
            "attachments": [{"type": "sticker", "payload": {}}],
        },
    }
    assert src.build_post_from_event(messaging) is None


def test_build_post_from_event_no_attachments_returns_none():
    src = _make_source()
    messaging = {
        "sender": {"id": "sender111"},
        "message": {"mid": "msg_abc", "text": "hello"},
    }
    assert src.build_post_from_event(messaging) is None


def test_build_reply_from_event_plain_text():
    src = _make_source()
    messaging = {
        "sender": {"id": "sender111"},
        "message": {"mid": "msg_xyz", "text": "Tokyo, Japan"},
    }
    reply = src.build_reply_from_event(messaging)
    assert isinstance(reply, TextReply)
    assert reply.item_id == "msg_xyz"
    assert reply.thread_id == "sender111"
    assert reply.text == "Tokyo, Japan"
    assert reply.reply_to_item_id is None


def test_build_reply_from_event_with_reply_to():
    src = _make_source()
    messaging = {
        "sender": {"id": "sender111"},
        "message": {
            "mid": "msg_xyz",
            "text": "Bali, Indonesia",
            "reply_to": {"mid": "ask_msg_123"},
        },
    }
    reply = src.build_reply_from_event(messaging)
    assert reply.reply_to_item_id == "ask_msg_123"


def test_build_reply_from_event_empty_text_returns_none():
    src = _make_source()
    messaging = {
        "sender": {"id": "sender111"},
        "message": {"mid": "msg_xyz"},
    }
    assert src.build_reply_from_event(messaging) is None


def _src():
    from bot.sources.instagram_api import InstagramAPISource
    return InstagramAPISource(access_token="t", ig_user_id="999", allowed_sender_id="111")


def _event(atype, url, reply_to_mid=None):
    message = {"mid": "msg_abc", "attachments": [{"type": atype, "payload": {"url": url}}]}
    if reply_to_mid:
        message["reply_to"] = {"mid": reply_to_mid}
    return {"sender": {"id": "111"}, "message": message}


def test_permalink_payload_yields_post_with_no_media():
    post = _src().build_post_from_event(_event("ig_reel", "https://www.instagram.com/reel/XYZ/"))
    assert post.link == "https://www.instagram.com/reel/XYZ/"
    assert post.media_url is None
    assert post.media_kind is None


def test_cdn_payload_yields_post_with_media_url():
    url = "https://lookaside.fbsbx.com/ig_messaging_cdn/?asset_id=1&signature=s"
    post = _src().build_post_from_event(_event("ig_reel", url))
    assert post.media_url == url
    assert post.media_kind == "video"


def test_share_attachment_leaves_media_kind_unknown():
    url = "https://lookaside.fbsbx.com/ig_messaging_cdn/?asset_id=2"
    post = _src().build_post_from_event(_event("share", url))
    assert post.media_url == url
    assert post.media_kind is None


def test_image_attachment_kind_is_image():
    url = "https://lookaside.fbsbx.com/ig_messaging_cdn/?asset_id=3"
    post = _src().build_post_from_event(_event("image", url))
    assert post.media_kind == "image"


def test_build_post_never_calls_instagrapi():
    """The whole point: no enrichment hook exists on this source any more."""
    src = _src()
    assert not hasattr(src, "_enrich")


def test_build_media_reply_from_event_carries_reply_to():
    url = "https://lookaside.fbsbx.com/ig_messaging_cdn/?asset_id=4"
    reply = _src().build_media_reply_from_event(_event("image", url, reply_to_mid="ask_1"))
    assert reply.media_url == url
    assert reply.reply_to_item_id == "ask_1"
    assert reply.thread_id == "111"


def test_build_media_reply_rejects_permalink():
    """A shared post is not an answer to an ask, even as a reply."""
    reply = _src().build_media_reply_from_event(
        _event("ig_reel", "https://www.instagram.com/reel/XYZ/", reply_to_mid="ask_1")
    )
    assert reply is None
