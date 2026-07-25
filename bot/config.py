"""Configuration loaded from environment / .env.

All runtime knobs live here so the rest of the code never reads os.environ
directly. Load order: a local .env file (if present) then real env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Real env vars take precedence."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    ig_username: str
    ig_password: str
    # IG user id (numeric, as a string) whose DMs we trust. Only shares from this
    # account are processed — keeps strangers from injecting data.
    allowed_sender_id: str
    model: str
    db_path: Path
    session_path: Path
    comments_limit: int
    comments_fetch_limit: int
    confidence_threshold: float
    ig_access_token: str | None = None
    ig_app_secret: str | None = None
    webhook_verify_token: str | None = None
    ig_user_id: str | None = None
    meta_app_id: str | None = None

    @staticmethod
    def load() -> "Config":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Validated here for a friendly early error; the genai client reads it from
        # the environment itself (GEMINI_API_KEY or GOOGLE_API_KEY).
        _require("GEMINI_API_KEY")
        return Config(
            ig_username=_require("IG_USERNAME"),
            ig_password=_require("IG_PASSWORD"),
            allowed_sender_id=_require("ALLOWED_SENDER_ID"),
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            db_path=Path(os.environ.get("DB_PATH", DATA_DIR / "db.sqlite")),
            session_path=Path(
                os.environ.get("IG_SESSION_PATH", DATA_DIR / "session.json")
            ),
            comments_limit=int(os.environ.get("COMMENTS_LIMIT", "25")),
            comments_fetch_limit=int(os.environ.get("COMMENTS_FETCH_LIMIT", "100")),
            confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.4")),
            ig_access_token=os.environ.get("IG_ACCESS_TOKEN"),
            ig_app_secret=os.environ.get("IG_APP_SECRET"),
            webhook_verify_token=os.environ.get("WEBHOOK_VERIFY_TOKEN"),
            ig_user_id=os.environ.get("IG_USER_ID"),
            meta_app_id=os.environ.get("META_APP_ID"),
        )


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value
