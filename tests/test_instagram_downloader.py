"""Offline tests for the no-login Instagram metadata reader."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from instagram_downloader import DownloadedComment, _select_comments, extract_shortcode, get_post_metadata


def test_extract_shortcode_accepts_post_and_reel_urls():
    assert extract_shortcode("https://www.instagram.com/p/ABC_12/") == "ABC_12"
    assert extract_shortcode("https://www.instagram.com/reels/XYZ-9/?x=1") == "XYZ-9"


def test_get_post_metadata_maps_caption_location_and_comments():
    creator_comment = SimpleNamespace(
        text="Location: Kyoto, Japan",
        owner=SimpleNamespace(username="creator"),
        likes_count=12,
        answers=iter(()),
    )
    post = SimpleNamespace(
        owner_username="creator",
        caption="Autumn in Kyoto",
        location=SimpleNamespace(name="Arashiyama"),
        get_comments=MagicMock(return_value=iter([creator_comment])),
        is_video=False,
        url="https://cdn.instagram.com/photo.jpg",
    )
    with patch("instagram_downloader.instaloader.Post.from_shortcode", return_value=post):
        metadata = get_post_metadata("https://www.instagram.com/p/ABC/", comments_limit=5)

    assert metadata.caption == "Autumn in Kyoto"
    assert metadata.location == "Arashiyama"
    assert metadata.comments[0].text == "Location: Kyoto, Japan"
    assert metadata.comments[0].is_creator is True
    assert metadata.comments[0].likes == 12
    assert metadata.media_url == "https://cdn.instagram.com/photo.jpg"
    assert metadata.media_kind == "image"


def test_comments_failure_keeps_caption_and_location():
    post = SimpleNamespace(
        owner_username="creator",
        caption="Autumn in Kyoto",
        location=SimpleNamespace(name="Arashiyama"),
        get_comments=MagicMock(side_effect=OSError("denied")),
        is_video=True,
        video_url="https://cdn.instagram.com/reel.mp4",
    )
    with patch("instagram_downloader.instaloader.Post.from_shortcode", return_value=post):
        metadata = get_post_metadata("https://www.instagram.com/p/ABC/")

    assert metadata.caption == "Autumn in Kyoto"
    assert metadata.location == "Arashiyama"
    assert metadata.comments == []
    assert metadata.media_url == "https://cdn.instagram.com/reel.mp4"
    assert metadata.media_kind == "video"


def test_get_post_metadata_includes_creator_replies_to_comments():
    creator_reply = SimpleNamespace(
        text="It's Lisbon, Portugal!",
        owner=SimpleNamespace(username="creator"),
        likes_count=4,
    )
    question = SimpleNamespace(
        text="Where is this?",
        owner=SimpleNamespace(username="visitor"),
        likes_count=1,
        answers=iter([creator_reply]),
    )
    post = SimpleNamespace(
        owner_username="creator",
        caption=None,
        location=None,
        get_comments=MagicMock(return_value=iter([question])),
        is_video=False,
        url="https://cdn.instagram.com/photo.jpg",
    )
    with patch("instagram_downloader.instaloader.Post.from_shortcode", return_value=post):
        metadata = get_post_metadata("https://www.instagram.com/p/ABC/")

    assert any(comment.is_creator and "Lisbon" in comment.text for comment in metadata.comments)


def test_comment_selection_keeps_a_buried_location_answer():
    comments = [DownloadedComment("😍", likes=10) for _ in range(10)]
    comments.append(DownloadedComment("Location: Miradouro de São Cristovão", likes=0))

    selected = _select_comments(comments, limit=3)

    assert any("Miradouro" in comment.text for comment in selected)
