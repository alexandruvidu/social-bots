#!/usr/bin/env python3

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import instaloader


_LOCATION_HINT_RE = re.compile(r"(?i)\blocation\b")
_CONNECTOR_WORDS = {"de", "da", "do", "du", "of", "van", "von", "di", "la", "le", "el", "the", "and"}


@dataclass(frozen=True)
class DownloadedComment:
    """The comment fields the destination pipeline can use."""

    text: str
    is_creator: bool = False
    likes: int = 0
    author: str | None = None


@dataclass(frozen=True)
class DownloadedPost:
    """Metadata fetched without logging an Instagram account in."""

    caption: str | None
    location: str | None
    comments: list[DownloadedComment]
    media_url: str | None = None
    media_kind: str | None = None


def _looks_like_place_name(text: str) -> bool:
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


def _select_comments(
    comments: list[DownloadedComment], limit: int
) -> list[DownloadedComment]:
    """Retain likely location answers instead of blindly keeping reactions."""
    if len(comments) <= limit:
        return comments
    priority = [
        comment for comment in comments
        if comment.is_creator or _looks_like_place_name(comment.text)
    ]
    rest = sorted(
        (comment for comment in comments if comment not in priority),
        key=lambda comment: comment.likes,
        reverse=True,
    )
    return (priority + rest)[:limit]


def extract_shortcode(url):
    """Extract an Instagram post/reel shortcode from a URL."""

    parsed = urlparse(url)
    path = parsed.path.strip("/")

    # Supported Instagram URL formats:
    # /p/SHORTCODE/
    # /reel/SHORTCODE/
    # /reels/SHORTCODE/
    # /tv/SHORTCODE/

    match = re.match(
        r"^(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)",
        path
    )

    if not match:
        raise ValueError(
            "Invalid Instagram URL.\n"
            "Expected something like:\n"
            "https://www.instagram.com/p/SHORTCODE/"
        )

    return match.group(1)


def _location_str(location) -> str | None:
    if not location:
        return None
    # Instaloader's PostLocation currently exposes name and slug.  Only the
    # human-readable name belongs in Gemini's prompt; keep this defensive so
    # a partial response still leaves the rest of the post usable.
    name = getattr(location, "name", None)
    return str(name) if name else None


def get_post_metadata(
    url: str, *, comments_limit: int = 25, comments_fetch_limit: int = 100
) -> DownloadedPost:
    """Fetch a public post's caption, geotag, and comments without login.

    This is deliberately a metadata reader, not a replacement for the
    webhook: the official Messaging API remains responsible for receiving and
    replying to DMs.  Instagram can deny anonymous requests (private post,
    login wall, rate limit), in which case callers should retain their
    media-only fallback rather than failing the incoming webhook event.
    """
    shortcode = extract_shortcode(url)
    loader = instaloader.Instaloader(
        quiet=True,
        download_comments=False,
        save_metadata=False,
    )
    post = instaloader.Post.from_shortcode(loader.context, shortcode)
    creator = post.owner_username
    comments: list[DownloadedComment] = []

    def add_comment(comment) -> None:
        """Map both top-level comments and replies into one trusted signal set."""
        text = getattr(comment, "text", None)
        if not text:
            return
        owner = getattr(comment, "owner", None)
        author = getattr(owner, "username", None)
        comments.append(
            DownloadedComment(
                text=str(text),
                is_creator=bool(creator) and author == creator,
                # `likes` is an iterator of profiles; `likes_count` is the
                # documented numeric count and does not need an account login.
                likes=int(getattr(comment, "likes_count", 0) or 0),
                author=str(author) if author else None,
            )
        )

    # get_comments is a separate request and is often the first part of an
    # anonymous lookup Instagram refuses.  Preserve caption/geotag if it does.
    try:
        for comment in post.get_comments():
            add_comment(comment)
            if len(comments) >= max(comments_fetch_limit, comments_limit):
                break
            # The useful "it's in X" answer is commonly a creator reply to a
            # user's question, not a top-level comment. Instaloader exposes
            # those replies via `answers`, so retain them under the same cap.
            for answer in getattr(comment, "answers", ()):
                add_comment(answer)
                if len(comments) >= max(comments_fetch_limit, comments_limit):
                    break
            if len(comments) >= max(comments_fetch_limit, comments_limit):
                break
    except (instaloader.exceptions.InstaloaderException, OSError):
        # Metadata is still useful, and the caller logs the enrichment result.
        pass

    is_video = bool(getattr(post, "is_video", False))
    media_url = getattr(post, "video_url", None) if is_video else getattr(post, "url", None)
    return DownloadedPost(
        caption=post.caption or None,
        location=_location_str(post.location),
        comments=_select_comments(comments, comments_limit),
        media_url=str(media_url) if media_url else None,
        media_kind="video" if is_video and media_url else "image" if media_url else None,
    )


def download_post(url, output_dir="downloads"):
    shortcode = extract_shortcode(url)

    print(f"Shortcode: {shortcode}")
    print("Loading Instagram post...")

    loader = instaloader.Instaloader(
        dirname_pattern=f"{output_dir}/{{profile}}",
        filename_pattern="{date_utc:%Y-%m-%d_%H-%M-%S}_{shortcode}",
        download_comments=True,
        save_metadata=True,
    )

    try:
        post = instaloader.Post.from_shortcode(
            loader.context,
            shortcode
        )

        print(f"Author: @{post.owner_username}")

        if post.typename == "GraphSidecar":
            print("Type: Carousel")
        elif post.is_video:
            print("Type: Video/Reel")
        else:
            print("Type: Photo")

        print("Downloading...")

        loader.download_post(
            post,
            target=post.owner_username
        )

        print("\nDownload complete.")

    except instaloader.exceptions.QueryReturnedNotFoundException:
        print("Post not found or is unavailable.")

    except instaloader.exceptions.LoginRequiredException:
        print("This post requires Instagram login.")

    except instaloader.exceptions.ConnectionException as e:
        print(f"Connection error: {e}")

    except instaloader.exceptions.PrivateProfileNotFollowedException:
        print("This is a private account that your session cannot access.")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 instagram_downloader.py <instagram_url>")
        sys.exit(1)

    url = sys.argv[1]

    try:
        download_post(url)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
