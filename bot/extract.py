"""Destination extraction via the Google Gemini API (free tier, structured output).

Three entry points, one `Extracted` result shape:

- `extract` — text only (geotag, caption, comments). Only the dormant
  `instagrapi` path can supply those, so the live pipeline rarely uses it.
- `analyze_media` — the live path: the media the official Messaging API webhook
  delivered, dispatched to video or image analysis by the attachment's declared
  kind, or by the download's Content-Type when the kind is unknown.
- `analyze_image` — a screenshot the user sent to answer a failed extraction.

Uses Gemini's JSON mode with a Pydantic response schema, so the model returns a
validated `Extracted` object directly. Reads GEMINI_API_KEY from the environment.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from .sources.base import PostComment

log = logging.getLogger(__name__)

# Free-tier Gemini returns 503 (overloaded) and 429 (rate limit) intermittently.
_RETRY_STATUSES = {429, 500, 503}
_MAX_ATTEMPTS = 4
# 429s whose suggested wait is longer than this aren't worth blocking the
# webhook request for; raise RateLimitedError instead so the caller can queue
# the post for a background retry.
_MAX_INLINE_WAIT = 5.0

# Shared across all three system prompts so a canonicalization rule can't
# drift out of sync between them. The destination string is the dedup key
# once it leaves this bot (matched by exact text downstream), so the same
# real-world place must always come back as the same characters.
_CANONICAL_DESTINATION = (
    "When you find one, normalize it to 'City, Country' (or 'Region, Country' / "
    "'Country' if that is the most specific that's clearly supported) using this "
    "EXACT canonical form every time, so the same real-world place always produces "
    "identical text across different posts: the place's most common English "
    "exonym (e.g. 'Rome' not 'Roma', 'Munich' not 'München'), the country's short "
    "official English name (e.g. 'Japan', 'United States' — not 'USA' or 'the "
    "States'), no diacritics, no leading articles ('the'), no administrative "
    "qualifiers beyond what's needed to identify the place (e.g. 'Kyoto, Japan' "
    "not 'Kyoto City, Kyoto Prefecture, Japan'), and exactly one comma with one "
    "space separating each part — no trailing punctuation or extra whitespace."
)

# Shared across all three system prompts for the same reason as
# _CANONICAL_DESTINATION: one instruction, applied identically everywhere,
# instead of three copies that can drift.
_TOPIC_INSTRUCTION = (
    "Also give a short topic phrase (roughly 3-8 words, under 60 characters) "
    "summarizing what this specific post/segment is actually about — this is used "
    "as a bookmark label, so it should tell someone what they'd get out of "
    "revisiting it, not just restate the destination or landmark name. Good "
    "examples: 'rules about trains in Japan', 'best ramen spots', 'how to skip "
    "the shrine queue'. Return null if nothing more specific than the "
    "destination/landmark itself is identifiable."
)


class RateLimitedError(Exception):
    """Gemini is quota-limited; retry after this many seconds."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Gemini rate limited; retry after {retry_after:.0f}s")

SYSTEM = (
    "You extract the travel destination(s) from a social media post. Most posts cover "
    "a single destination, but some mention more than one (e.g. a multi-city trip "
    "recap) — find every distinct one. "
    "You are given labelled sections: a geotag/location, the caption, and comments. "
    "Trust signals in this order:\n"
    "1. The geotag, when present and specific.\n"
    "2. The caption (written by the creator).\n"
    "3. Comments written by the CREATOR (these are marked) — e.g. the creator "
    "replying 'it's X' to a 'where is this?' question. Treat these as reliable.\n"
    "4. Other users' comments are UNRELIABLE — random people guess, joke, or give "
    "wrong answers. Do not trust a non-creator comment's answer unless it is "
    "strongly corroborated (e.g. several independent comments agree, or it has many "
    "likes). If the only place name comes from a single unverified non-creator "
    "comment, either return null or give a low confidence (< 0.4).\n"
    "Put the single most prominent destination in the top-level destination field "
    "(with its own landmark/place_type/source_field/confidence). If the post clearly "
    "covers OTHER distinct destinations too, add one entry per extra destination to "
    "more_places, using the same fields and trust rules; leave more_places empty if "
    "there's only one destination.\n"
    "If the post instead lists several specific named landmarks WITHIN one "
    "destination (e.g. an itinerary like 'Day 1: Pico do Arieiro, Fanal Forest, "
    "Seixal Beach' all in Madeira), do not pick just one — add one entry per "
    "landmark to more_places, each repeating the same destination but with its "
    "own landmark/place_type/source_field/confidence, so every named stop is "
    "captured.\n"
    "Return null for a destination if no real-world place is reliably identifiable. "
    + _CANONICAL_DESTINATION + " "
    "Also extract the specific named place shown or mentioned within that destination, "
    "if any is identifiable — this can be a tourist landmark/attraction (e.g. 'Eiffel "
    "Tower', 'Hongya Cave'), a specific restaurant/food stall (e.g. 'Chongqing BBQ'), "
    "or a specific hotel. Use the same trust rules for it as for the destination. Return "
    "null for landmark if the post is about a destination generally with no specific "
    "named site. When a landmark is found, set place_type to one of 'landmark' "
    "(tourist attraction/sight), 'restaurant' (any food/drink venue), or 'hotel' "
    "(accommodation) — whichever best describes that place; null if landmark is null. "
    "Set source_field to where the destination came from ('location', 'caption', or "
    "'comments'), and confidence to a 0-1 score reflecting how trustworthy the source was. "
    + _TOPIC_INSTRUCTION
)


VIDEO_SYSTEM = (
    "You extract the travel destination(s) from a video. Most videos cover a single "
    "destination, but some cover more than one (e.g. a multi-city trip recap) — find "
    "every distinct one. Watch the entire clip and look for: "
    "1. On-screen text overlays, captions, or location tags naming a place. "
    "2. Spoken words mentioning a location. "
    "3. Recognizable visual landmarks, landscapes, or signs. "
    "Put the single most prominent destination in the top-level destination field "
    "(with its own landmark/place_type/source_field/confidence). If the video clearly "
    "covers OTHER distinct destinations too, add one entry per extra destination to "
    "more_places, using the same fields; leave more_places empty if there's only one "
    "destination. "
    "Return null for a destination if no real-world place is reliably identifiable. "
    + _CANONICAL_DESTINATION + " "
    "Also extract the specific named place shown or mentioned — a tourist landmark/ "
    "attraction (e.g. 'Eiffel Tower', 'Hongya Cave'), a specific restaurant/food stall "
    "(e.g. 'Chongqing BBQ'), or a specific hotel — or null if none is identifiable. "
    "When a landmark is found, set place_type to one of 'landmark', 'restaurant', or "
    "'hotel' — whichever best describes that place; null if landmark is null. "
    "Set source_field to 'video', and confidence to a 0-1 score. "
    + _TOPIC_INSTRUCTION
)


SCREENSHOT_SYSTEM = (
    "You extract the travel destination(s) from a SCREENSHOT of an Instagram post "
    "or reel. This is a capture of a phone screen, so the app's own interface is "
    "visible around the content. Read it in this order:\n"
    "1. Any location/geotag label shown under the account name at the top — this "
    "is the most reliable signal when present.\n"
    "2. The caption text. It is often truncated with '... more', so use whatever "
    "is legible and don't guess at the hidden part.\n"
    "3. Text overlays burned into the media itself.\n"
    "4. Visible comment text, which is UNRELIABLE unless it is written by the "
    "account that posted (the same username shown at the top).\n"
    "5. Recognizable visual landmarks, landscapes, or signs in the image.\n"
    "Ignore Instagram's interface chrome — like/comment/share counts, button "
    "labels, the navigation bar, the status bar, and the search field. "
    "Put the single most prominent destination in the top-level destination field "
    "(with its own landmark/place_type/source_field/confidence). If the screenshot "
    "clearly shows OTHER distinct destinations too, add one entry per extra "
    "destination to more_places, using the same fields; leave more_places empty if "
    "there's only one destination. "
    "Return null for a destination if no real-world place is reliably identifiable. "
    + _CANONICAL_DESTINATION + " "
    "Also extract the specific named place shown or mentioned — a tourist landmark/ "
    "attraction (e.g. 'Eiffel Tower', 'Hongya Cave'), a specific restaurant/food "
    "stall (e.g. 'Chongqing BBQ'), or a specific hotel — or null if none is "
    "identifiable. When a landmark is found, set place_type to one of 'landmark', "
    "'restaurant', or 'hotel'; null if landmark is null. "
    "Set source_field to 'screenshot', and confidence to a 0-1 score. "
    + _TOPIC_INSTRUCTION
)


class Place(BaseModel):
    destination: Optional[str] = None  # "City, Country", or None if none found
    landmark: Optional[str] = None     # specific named site/attraction, or None
    place_type: Optional[str] = None   # "landmark" | "restaurant" | "hotel", or None
    confidence: float = 0.0            # 0-1
    source_field: Optional[str] = None  # "location" | "caption" | "comments"
    topic: Optional[str] = None        # short bookmark-style label, e.g. "rules about trains in Japan"


class Extracted(Place):
    more_places: list[Place] = []      # any other distinct destinations found


def _fmt_comment(c: PostComment) -> str:
    who = "CREATOR" if c.is_creator else "other"
    likes = f", {c.likes} likes" if c.likes else ""
    return f"- [{who}{likes}] {c.text}"


def build_prompt(
    caption: str | None,
    location: str | None,
    comments: Sequence[PostComment],
) -> str:
    parts: list[str] = []
    parts.append(f"LOCATION (geotag): {location or '(none)'}")
    parts.append(f"CAPTION: {caption or '(none)'}")
    if comments:
        # Creator comments first so the trustworthy signal is prominent.
        ordered = sorted(comments, key=lambda c: (not c.is_creator, -c.likes))
        joined = "\n".join(_fmt_comment(c) for c in ordered)
        parts.append(f"COMMENTS (each tagged CREATOR or other):\n{joined}")
    else:
        parts.append("COMMENTS: (none)")
    return "\n\n".join(parts)


def extract(
    caption: str | None,
    location: str | None,
    comments: Sequence[PostComment],
    *,
    model: str = "gemini-2.5-flash",
    client: "genai.Client | None" = None,
) -> Extracted:
    client = client or genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    contents = build_prompt(caption, location, comments)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=Extracted,
    )
    resp = _generate_with_retry(client, model, contents, config)
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, Extracted):
        return parsed
    return Extracted()


# Fallback when the server doesn't tell us the image type. Gemini accepts
# JPEG/PNG/WebP; JPEG is what Instagram screenshots almost always are.
_IMAGE_MIME_DEFAULT = "image/jpeg"
_VIDEO_MIME = "video/mp4"


def _analyze_video_file(path: Path, *, model: str, client: "genai.Client") -> Extracted:
    """Run video analysis on an already-downloaded file.

    Split out from analyze_video so analyze_media can hand over a file it has
    already fetched instead of downloading the same media a second time.
    """
    uploaded = client.files.upload(
        file=path,
        config=types.UploadFileConfig(mime_type=_VIDEO_MIME),
    )
    try:
        _wait_for_active(client, uploaded.name)
        video_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type=_VIDEO_MIME)
        config = types.GenerateContentConfig(
            system_instruction=VIDEO_SYSTEM,
            response_mime_type="application/json",
            response_schema=Extracted,
        )
        resp = _generate_with_retry(client, model, [video_part], config)
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, Extracted):
            return parsed
        return Extracted()
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            log.warning("Failed to delete Gemini file %s", uploaded.name, exc_info=True)


def _analyze_image_file(
    path: Path, content_type: str, *, model: str, client: "genai.Client"
) -> Extracted:
    """Run image analysis on an already-downloaded file, inline (no Files API)."""
    mime = (
        content_type
        if isinstance(content_type, str) and content_type.startswith("image/")
        else _IMAGE_MIME_DEFAULT
    )
    part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
    config = types.GenerateContentConfig(
        system_instruction=SCREENSHOT_SYSTEM,
        response_mime_type="application/json",
        response_schema=Extracted,
    )
    resp = _generate_with_retry(client, model, [part], config)
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, Extracted):
        return parsed
    return Extracted()


def analyze_video(
    video_url: str,
    *,
    model: str = "gemini-2.5-flash",
    client: "genai.Client | None" = None,
) -> Extracted:
    client = client or genai.Client()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as _tmp:
        tmp = Path(_tmp.name)
    try:
        _download_media(video_url, tmp)
        return _analyze_video_file(tmp, model=model, client=client)
    finally:
        tmp.unlink(missing_ok=True)


def analyze_image(
    image_url: str,
    *,
    model: str = "gemini-2.5-flash",
    client: "genai.Client | None" = None,
) -> Extracted:
    """Identify a destination from a single image (typically a screenshot).

    Unlike analyze_video this passes bytes inline rather than going through the
    Files API — no upload, no ACTIVE-state polling — so a screenshot reply
    resolves in one round trip.
    """
    client = client or genai.Client()
    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as _tmp:
        tmp = Path(_tmp.name)
    try:
        content_type = _download_media(image_url, tmp)
        return _analyze_image_file(tmp, content_type, model=model, client=client)
    finally:
        tmp.unlink(missing_ok=True)


def analyze_media(
    media_url: str,
    kind: str | None = None,
    *,
    model: str = "gemini-2.5-flash",
    client: "genai.Client | None" = None,
) -> Extracted:
    """Dispatch to image or video analysis, resolving the type when unknown."""
    if kind == "image":
        return analyze_image(media_url, model=model, client=client)
    if kind == "video":
        return analyze_video(media_url, model=model, client=client)
    # Unknown kind — Instagram's `share` type covers ordinary photo posts as
    # well as reels. Download once and let the response's own Content-Type
    # decide. A separate HEAD sniff is not an option: signed CDN URLs commonly
    # reject HEAD with 403/405, and treating that failure as "video" sends
    # photo posts to the Files API tagged video/mp4, which just fails.
    client = client or genai.Client()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as _tmp:
        tmp = Path(_tmp.name)
    try:
        content_type = _download_media(media_url, tmp)
        if content_type.startswith("image/"):
            return _analyze_image_file(tmp, content_type, model=model, client=client)
        if not content_type.startswith("video/"):
            # Reels dominate the shares this bot sees, so an unrecognised type
            # is far likelier to be a video than an image.
            log.warning(
                "Unrecognised Content-Type %r for %s; treating as video.",
                content_type, media_url,
            )
        return _analyze_video_file(tmp, model=model, client=client)
    finally:
        tmp.unlink(missing_ok=True)


def _retry_delay_seconds(e: "genai_errors.APIError") -> Optional[float]:
    """Pull the server-suggested retry delay (RetryInfo.retryDelay) out of a 429."""
    try:
        details = e.details.get("error", {}).get("details", [])
    except AttributeError:
        return None
    for d in details:
        if d.get("@type", "").endswith("RetryInfo"):
            raw = d.get("retryDelay", "")
            if raw.endswith("s"):
                try:
                    return float(raw[:-1])
                except ValueError:
                    return None
    return None


def _generate_with_retry(client, model, contents, config):
    delay = 2.0
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except genai_errors.APIError as e:
            status = getattr(e, "code", None)
            if status == 429:
                wait = _retry_delay_seconds(e) or 60.0
                if wait > _MAX_INLINE_WAIT:
                    # Long quota wait: don't block this request for it, let
                    # the caller queue the post for a background retry.
                    raise RateLimitedError(wait) from e
                if attempt < _MAX_ATTEMPTS:
                    log.warning(
                        "Gemini 429 (attempt %d/%d); retrying in %.0fs",
                        attempt, _MAX_ATTEMPTS, wait,
                    )
                    time.sleep(wait)
                    continue
                raise RateLimitedError(wait) from e
            if status in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS:
                log.warning(
                    "Gemini %s (attempt %d/%d); retrying in %.0fs",
                    status, attempt, _MAX_ATTEMPTS, delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            raise


def _download_media(url: str, dest: Path) -> str:
    """Stream `url` to `dest`. Returns the response Content-Type (may be "").

    The type is returned because Instagram's `share` attachment type doesn't
    say whether the media is an image or a video — the server does.
    """
    import httpx
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").split(";")[0].strip().lower()
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
    return content_type


def _wait_for_active(client: "genai.Client", file_name: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        f = client.files.get(name=file_name)
        if f.state == types.FileState.ACTIVE:
            return
        if f.state == types.FileState.FAILED:
            raise RuntimeError(f"Gemini file processing failed: {file_name}")
        time.sleep(2.0)
    raise TimeoutError(f"Gemini file not ACTIVE after {timeout}s: {file_name}")


def is_success(result: Place, threshold: float) -> bool:
    return bool(result.destination) and result.confidence >= threshold
