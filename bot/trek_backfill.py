"""Manual reconciliation: push every already-saved destination into TREK.

Not part of the live path — run by hand if a live push ever failed and you
want to check/fix drift between data/db.sqlite and TREK:

    python -m bot.trek_backfill

Re-running this is safe: push_destination derives "already synced" from
TREK's own data (see bot/trek/logic.py), so an already-pushed destination
is a no-op, not a duplicate.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from typing import Any

from .config import Config
from .trek import push_destination

log = logging.getLogger("trek_backfill")


def read_destinations(db_path: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT platform, link, destination, landmark, place_type, caption_snippet "
            "FROM destinations WHERE destination IS NOT NULL ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def backfill(cfg: Config) -> dict[str, int]:
    stats = {"pushed": 0, "failed": 0}
    for row in read_destinations(str(cfg.db_path)):
        ok = push_destination(
            cfg,
            platform=row["platform"],
            link=row["link"],
            destination=row["destination"],
            landmark=row["landmark"],
            place_type=row["place_type"],
            topic=None,  # historical rows predate the `topic` field
            caption_snippet=row["caption_snippet"],
        )
        stats["pushed" if ok else "failed"] += 1
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.load()
    if not (cfg.trek_url and cfg.trek_api_token):
        print("TREK_URL/TREK_API_TOKEN not set — nothing to do.", file=sys.stderr)
        return 1
    stats = backfill(cfg)
    print(f"Backfill complete: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
