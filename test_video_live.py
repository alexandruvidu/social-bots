"""One-shot test: fetch the pending reel's video URL and run analyze_video() on it."""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

sys.path.insert(0, str(Path(__file__).parent))

from bot.config import Config
from bot.extract import analyze_video, is_success
from bot.sources.instagram import login_client

cfg = Config.load()
DATA_DIR = Path(__file__).parent / "data"

print("Logging into Instagram (using saved session)...")
client = login_client(cfg.ig_username, cfg.ig_password, cfg.session_path)

SHORTCODE = "DTfwOTZEvDx"
print(f"\nFetching media info for reel {SHORTCODE}...")
try:
    pk = client.media_pk_from_code(SHORTCODE)
    media = client.media_info(pk)
    video_url = str(media.video_url) if getattr(media, "video_url", None) else None
    print(f"  caption: {getattr(media, 'caption_text', None)!r}")
    print(f"  location: {getattr(media, 'location', None)}")
    print(f"  video_url: {(video_url[:80] + '...') if video_url and len(video_url) > 80 else video_url!r}")
except Exception as e:
    print(f"  media_info failed: {e}")
    video_url = None

if not video_url:
    print("\nNo video URL available — cannot test analyze_video().")
    sys.exit(1)

print(f"\nRunning analyze_video()...")
result = analyze_video(video_url, model=cfg.model)
print(f"\nResult:")
print(f"  destination : {result.destination!r}")
print(f"  confidence  : {result.confidence:.2f}")
print(f"  source_field: {result.source_field!r}")
print(f"  is_success  : {is_success(result, cfg.confidence_threshold)}")
