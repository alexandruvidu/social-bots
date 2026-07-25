"""Extraction tests.

The prompt and fake-client tests run offline. `test_live_extraction` hits the real
Gemini API and is skipped unless GEMINI_API_KEY is set and RUN_LIVE_TESTS=1, so the
default test run needs no network.
"""
import os

import pytest

from bot.extract import Extracted, build_prompt, extract, is_success
from bot.sources.base import PostComment


class _FakeResp:
    def __init__(self, parsed):
        self.parsed = parsed


class _FakeModels:
    def __init__(self, parsed):
        self._parsed = parsed

    def generate_content(self, **_kwargs):
        return _FakeResp(self._parsed)


class _FakeClient:
    def __init__(self, parsed):
        self.models = _FakeModels(parsed)


def test_build_prompt_labels_sections():
    p = build_prompt(
        "Loving Lisbon",
        "Lisbon, Portugal",
        [PostComment(text="best pasteis!", is_creator=False)],
    )
    assert "LOCATION (geotag): Lisbon, Portugal" in p
    assert "CAPTION: Loving Lisbon" in p
    assert "COMMENTS" in p
    assert "best pasteis!" in p


def test_build_prompt_handles_missing():
    p = build_prompt(None, None, [])
    assert "LOCATION (geotag): (none)" in p
    assert "CAPTION: (none)" in p
    assert "COMMENTS: (none)" in p


def test_build_prompt_tags_and_orders_creator_first():
    p = build_prompt(
        None,
        None,
        [
            PostComment(text="is it Rome?", is_creator=False, likes=2),
            PostComment(text="it's Matera, Italy!", is_creator=True),
        ],
    )
    creator_idx = p.index("Matera")
    other_idx = p.index("is it Rome?")
    assert creator_idx < other_idx  # creator comment listed first
    assert "[CREATOR]" in p
    assert "[other, 2 likes]" in p


def test_extract_uses_client_parsed_output():
    fake = _FakeClient(Extracted(destination="Bali, Indonesia", confidence=0.8,
                                 source_field="caption"))
    result = extract("week in bali", None, [], client=fake)
    assert result.destination == "Bali, Indonesia"
    assert is_success(result, 0.4) is True


def test_extract_failure_below_threshold():
    fake = _FakeClient(Extracted(destination=None, confidence=0.0))
    result = extract("just a cool sunset", None, [], client=fake)
    assert is_success(result, 0.4) is False


def test_extract_none_parsed_output():
    fake = _FakeClient(None)
    result = extract("x", None, [], client=fake)
    assert result.destination is None
    assert is_success(result, 0.4) is False


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") and os.environ.get("RUN_LIVE_TESTS")),
    reason="live API test; set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run",
)
def test_live_extraction():
    # caption-only
    r1 = extract("Spent 5 incredible days exploring Kyoto 🇯🇵", None, [])
    assert r1.destination and "Kyoto" in r1.destination

    # location-only
    r2 = extract(None, "Santorini, Greece", [])
    assert r2.destination and "Santorini" in r2.destination

    # creator comment answers a "where is this?" question -> trusted
    r3 = extract(
        "where is this?? 😍",
        None,
        [PostComment(text="It's Hallstatt, Austria!", is_creator=True)],
    )
    assert r3.destination and "Hallstatt" in r3.destination

    # only a random (non-creator) guess -> low confidence / failure
    r4 = extract(
        "where is this?? 😍",
        None,
        [PostComment(text="looks like Switzerland lol", is_creator=False)],
    )
    assert is_success(r4, 0.4) is False

    # no destination at all
    r5 = extract("my cat being cute", None, [PostComment(text="so fluffy")])
    assert is_success(r5, 0.4) is False


from unittest.mock import MagicMock, patch

from bot.extract import analyze_video


class _FakeFile:
    def __init__(self, name="files/abc123", state="ACTIVE", uri="https://cdn/files/abc123"):
        self.name = name
        self.state = state
        self.uri = uri


class _FakeFiles:
    def __init__(self, upload_file, get_file=None):
        self._upload_file = upload_file
        self._get_file = get_file or upload_file

    def upload(self, *, file, config=None):
        return self._upload_file

    def get(self, *, name):
        return self._get_file

    def delete(self, *, name):
        pass


class _FakeClientForVideo:
    def __init__(self, parsed, upload_file=None, get_file=None):
        active_file = upload_file or _FakeFile()
        self.files = _FakeFiles(active_file, get_file)
        self.models = _FakeModels(parsed)


def _mock_stream():
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.iter_bytes.return_value = [b"fake-video-data"]
    return mock_resp


def test_analyze_video_returns_extracted():
    fake = _FakeClientForVideo(
        parsed=Extracted(destination="Kyoto, Japan", confidence=0.9, source_field="video"),
    )
    with patch("httpx.stream", return_value=_mock_stream()):
        result = analyze_video("https://example.com/video.mp4", client=fake)
    assert result.destination == "Kyoto, Japan"
    assert result.confidence == 0.9
    assert result.source_field == "video"


def test_analyze_video_deletes_file_on_success():
    active_file = _FakeFile()
    fake = _FakeClientForVideo(
        parsed=Extracted(destination="Rome, Italy", confidence=0.85, source_field="video"),
        upload_file=active_file,
    )
    deleted = []
    fake.files.delete = lambda *, name: deleted.append(name)
    with patch("httpx.stream", return_value=_mock_stream()):
        analyze_video("https://example.com/video.mp4", client=fake)
    assert active_file.name in deleted


def test_analyze_video_deletes_file_on_generate_failure():
    active_file = _FakeFile()

    class _BrokenModels:
        def generate_content(self, **_):
            raise RuntimeError("Gemini exploded")

    fake = _FakeClientForVideo(parsed=None, upload_file=active_file)
    fake.models = _BrokenModels()
    deleted = []
    fake.files.delete = lambda *, name: deleted.append(name)
    with patch("httpx.stream", return_value=_mock_stream()):
        with pytest.raises(RuntimeError, match="Gemini exploded"):
            analyze_video("https://example.com/video.mp4", client=fake)
    assert active_file.name in deleted


def test_analyze_video_none_parsed_returns_empty():
    fake = _FakeClientForVideo(parsed=None)
    with patch("httpx.stream", return_value=_mock_stream()):
        result = analyze_video("https://example.com/video.mp4", client=fake)
    assert result.destination is None
