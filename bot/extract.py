"""Destination extraction via the Google Gemini API (free tier, structured output).

Feeds the post's geotag, caption, and top comments to Gemini and gets back a
single normalized destination (or null when none is identifiable).

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
    "When you do find one, normalize it to 'City, Country' (or 'Region, Country' / "
    "'Country' if that is the most specific that's clearly supported). "
    "Also extract the specific named place shown or mentioned within that destination, "
    "if any is identifiable — this can be a tourist landmark/attraction (e.g. 'Eiffel "
    "Tower', 'Hongya Cave'), a specific restaurant/food stall (e.g. 'Chongqing BBQ'), "
    "or a specific hotel. Use the same trust rules for it as for the destination. Return "
    "null for landmark if the post is about a destination generally with no specific "
    "named site. When a landmark is found, set place_type to one of 'landmark' "
    "(tourist attraction/sight), 'restaurant' (any food/drink venue), or 'hotel' "
    "(accommodation) — whichever best describes that place; null if landmark is null. "
    "Set source_field to where the destination came from ('location', 'caption', or "
    "'comments'), and confidence to a 0-1 score reflecting how trustworthy the source was."
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
    "When you find one, normalize to 'City, Country' (or 'Region, Country' / "
    "'Country' if more specific is unclear). "
    "Also extract the specific named place shown or mentioned — a tourist landmark/ "
    "attraction (e.g. 'Eiffel Tower', 'Hongya Cave'), a specific restaurant/food stall "
    "(e.g. 'Chongqing BBQ'), or a specific hotel — or null if none is identifiable. "
    "When a landmark is found, set place_type to one of 'landmark', 'restaurant', or "
    "'hotel' — whichever best describes that place; null if landmark is null. "
    "Set source_field to 'video', and confidence to a 0-1 score."
)


class Place(BaseModel):
    destination: Optional[str] = None  # "City, Country", or None if none found
    landmark: Optional[str] = None     # specific named site/attraction, or None
    place_type: Optional[str] = None   # "landmark" | "restaurant" | "hotel", or None
    confidence: float = 0.0            # 0-1
    source_field: Optional[str] = None  # "location" | "caption" | "comments"


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
        _download_video(video_url, tmp)
        uploaded = client.files.upload(
            file=tmp,
            config=types.UploadFileConfig(mime_type="video/mp4"),
        )
        try:
            _wait_for_active(client, uploaded.name)
            video_part = types.Part.from_uri(
                file_uri=uploaded.uri, mime_type="video/mp4"
            )
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


def _download_video(url: str, dest: Path) -> None:
    import httpx
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)


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
