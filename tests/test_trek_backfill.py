import sqlite3
from unittest.mock import patch

from bot.trek_backfill import backfill, read_destinations


def _make_test_db(tmp_path, rows):
    db_path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE destinations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            link TEXT NOT NULL,
            destination TEXT,
            landmark TEXT,
            place_type TEXT,
            caption_snippet TEXT
        )"""
    )
    conn.executemany(
        "INSERT INTO destinations (platform, link, destination, landmark, place_type, caption_snippet) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db_path)


def test_read_destinations_returns_rows_with_a_destination(tmp_path):
    db_path = _make_test_db(
        tmp_path,
        [
            ("instagram", "https://instagram.com/p/1", "Kyoto, Japan", None, None, "so pretty"),
            ("instagram", "https://instagram.com/p/2", None, None, None, None),
        ],
    )
    rows = read_destinations(db_path)
    assert len(rows) == 1
    assert rows[0]["destination"] == "Kyoto, Japan"
    assert rows[0]["link"] == "https://instagram.com/p/1"


def test_read_destinations_is_read_only(tmp_path):
    db_path = _make_test_db(
        tmp_path, [("instagram", "https://instagram.com/p/1", "Kyoto, Japan", None, None, None)]
    )
    read_destinations(db_path)
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM destinations").fetchone()[0] == 1
    conn.close()


def test_backfill_pushes_every_row_and_counts_results(tmp_path):
    db_path = _make_test_db(
        tmp_path,
        [
            ("instagram", "https://instagram.com/p/1", "Kyoto, Japan", "Fushimi Inari", "landmark", "wow"),
            ("tiktok", "https://tiktok.com/@x/2", "Paris, France", None, None, None),
        ],
    )

    class FakeCfg:
        trek_url = "http://trek.local:3000"
        trek_api_token = "trek_abc"
        db_path_str = db_path

    cfg = FakeCfg()
    cfg.db_path = db_path

    with patch("bot.trek_backfill.push_destination", side_effect=[True, False]) as mock_push:
        stats = backfill(cfg)

    assert mock_push.call_count == 2
    first_call_kwargs = mock_push.call_args_list[0].kwargs
    assert first_call_kwargs["destination"] == "Kyoto, Japan"
    assert first_call_kwargs["landmark"] == "Fushimi Inari"
    assert first_call_kwargs["topic"] is None  # historical rows predate the `topic` field
    assert stats == {"pushed": 1, "failed": 1}
