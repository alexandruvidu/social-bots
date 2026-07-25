"""SQLite storage: destinations, processed-item tracking, and pending lookups.

The connection is opened per Store instance. Pass ":memory:" as the path in tests.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS destinations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    platform        TEXT NOT NULL,
    link            TEXT NOT NULL,
    destination     TEXT NOT NULL,
    landmark        TEXT,
    place_type      TEXT,
    confidence      REAL,
    source_field    TEXT,
    caption_snippet TEXT,
    sender          TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed (
    platform   TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (platform, item_id)
);

CREATE TABLE IF NOT EXISTS pending (
    platform        TEXT NOT NULL,
    thread_id       TEXT NOT NULL,
    link            TEXT NOT NULL,
    caption_snippet TEXT,
    ask_msg_id      TEXT,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (platform, thread_id)
);

CREATE TABLE IF NOT EXISTS retry_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    platform        TEXT NOT NULL,
    item_id         TEXT NOT NULL,
    payload         TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: Path | str):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        # Migration: add ask_msg_id column if the DB predates it.
        try:
            self.conn.execute("ALTER TABLE pending ADD COLUMN ask_msg_id TEXT")
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_ask "
                "ON pending (platform, ask_msg_id) WHERE ask_msg_id IS NOT NULL"
            )
            self.conn.commit()
        except Exception:  # noqa: BLE001
            pass  # column already exists
        # Migration: add landmark/place_type columns if the DB predates them.
        for ddl in (
            "ALTER TABLE destinations ADD COLUMN landmark TEXT",
            "ALTER TABLE destinations ADD COLUMN place_type TEXT",
        ):
            try:
                self.conn.execute(ddl)
                self.conn.commit()
            except Exception:  # noqa: BLE001
                pass  # column already exists
        self._migrate_destinations_unique_key()
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_destinations_unique "
            "ON destinations (link, destination, COALESCE(landmark, ''))"
        )
        self.conn.commit()

    def _migrate_destinations_unique_key(self) -> None:
        """Rebuild `destinations` if it still has an old inline UNIQUE(link) or
        UNIQUE(link, destination) constraint. Those predate per-landmark rows
        (a single destination can now have several landmark entries from the
        same post), so any inline UNIQUE must go — dedup now lives solely in
        the idx_destinations_unique expression index created by SCHEMA."""
        for row in self.conn.execute("PRAGMA index_list(destinations)").fetchall():
            if not row["unique"] or row["name"] == "idx_destinations_unique":
                continue
            cols = [
                c["name"]
                for c in self.conn.execute(f"PRAGMA index_info({row['name']})").fetchall()
            ]
            if cols in (["link"], ["link", "destination"]):
                self.conn.executescript(
                    """
                    CREATE TABLE destinations_new (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        platform        TEXT NOT NULL,
                        link            TEXT NOT NULL,
                        destination     TEXT NOT NULL,
                        landmark        TEXT,
                        place_type      TEXT,
                        confidence      REAL,
                        source_field    TEXT,
                        caption_snippet TEXT,
                        sender          TEXT,
                        created_at      TEXT NOT NULL
                    );
                    INSERT INTO destinations_new
                        (id, platform, link, destination, landmark, place_type,
                         confidence, source_field, caption_snippet, sender, created_at)
                    SELECT id, platform, link, destination, landmark, place_type,
                           confidence, source_field, caption_snippet, sender, created_at
                    FROM destinations;
                    DROP TABLE destinations;
                    ALTER TABLE destinations_new RENAME TO destinations;
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_destinations_unique
                        ON destinations (link, destination, COALESCE(landmark, ''));
                    """
                )
                self.conn.commit()
                return

    def close(self) -> None:
        self.conn.close()

    # --- destinations ---
    def save_destination(
        self,
        *,
        platform: str,
        link: str,
        destination: str,
        landmark: str | None = None,
        place_type: str | None = None,
        confidence: float | None,
        source_field: str | None,
        caption_snippet: str | None,
        sender: str | None,
    ) -> bool:
        """Insert a destination. Returns False if the link already exists (dedup)."""
        try:
            self.conn.execute(
                """INSERT INTO destinations
                   (platform, link, destination, landmark, place_type, confidence,
                    source_field, caption_snippet, sender, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    platform,
                    link,
                    destination,
                    landmark,
                    place_type,
                    confidence,
                    source_field,
                    caption_snippet,
                    sender,
                    _now(),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate link

    def get_destinations_for_link(self, platform: str, link: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM destinations WHERE platform = ? AND link = ? ORDER BY id",
            (platform, link),
        ).fetchall()

    # --- processed tracking ---
    def is_processed(self, platform: str, item_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processed WHERE platform = ? AND item_id = ?",
            (platform, item_id),
        ).fetchone()
        return row is not None

    def mark_processed(self, platform: str, item_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO processed (platform, item_id, created_at) "
            "VALUES (?, ?, ?)",
            (platform, item_id, _now()),
        )
        self.conn.commit()

    # --- pending (failed extraction awaiting a user reply) ---
    def add_pending(
        self,
        *,
        platform: str,
        thread_id: str,
        link: str,
        caption_snippet: str | None,
        ask_msg_id: str | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO pending
               (platform, thread_id, link, caption_snippet, ask_msg_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (platform, thread_id, link, caption_snippet, ask_msg_id, _now()),
        )
        self.conn.commit()

    def get_pending(self, platform: str, thread_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pending WHERE platform = ? AND thread_id = ?",
            (platform, thread_id),
        ).fetchone()

    def get_pending_by_ask_msg(self, platform: str, ask_msg_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pending WHERE platform = ? AND ask_msg_id = ?",
            (platform, ask_msg_id),
        ).fetchone()

    def clear_pending(self, platform: str, thread_id: str) -> None:
        self.conn.execute(
            "DELETE FROM pending WHERE platform = ? AND thread_id = ?",
            (platform, thread_id),
        )
        self.conn.commit()

    def clear_pending_by_ask_msg(self, platform: str, ask_msg_id: str) -> None:
        self.conn.execute(
            "DELETE FROM pending WHERE platform = ? AND ask_msg_id = ?",
            (platform, ask_msg_id),
        )
        self.conn.commit()

    def enqueue_retry(self, platform: str, item_id: str, payload: str, retry_after: float) -> None:
        next_attempt = (datetime.now(timezone.utc) + timedelta(seconds=retry_after)).isoformat()
        self.conn.execute(
            "INSERT INTO retry_queue (platform, item_id, payload, attempts, next_attempt_at, created_at) "
            "VALUES (?, ?, ?, 0, ?, ?)",
            (platform, item_id, payload, next_attempt, _now()),
        )
        self.conn.commit()

    def due_retries(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM retry_queue WHERE next_attempt_at <= ? ORDER BY id", (_now(),)
        ).fetchall()

    def reschedule_retry(self, row_id: int, retry_after: float) -> None:
        next_attempt = (datetime.now(timezone.utc) + timedelta(seconds=retry_after)).isoformat()
        self.conn.execute(
            "UPDATE retry_queue SET attempts = attempts + 1, next_attempt_at = ? WHERE id = ?",
            (next_attempt, row_id),
        )
        self.conn.commit()

    def delete_retry(self, row_id: int) -> None:
        self.conn.execute("DELETE FROM retry_queue WHERE id = ?", (row_id,))
        self.conn.commit()
