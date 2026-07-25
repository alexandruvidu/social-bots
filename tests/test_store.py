"""Storage tests — pure SQLite, no network."""
import sqlite3

from bot.store import Store


def make_store() -> Store:
    return Store(":memory:")


def test_save_and_dedup():
    s = make_store()
    ok = s.save_destination(
        platform="instagram",
        link="https://insta/p/ABC/",
        destination="Kyoto, Japan",
        confidence=0.9,
        source_field="caption",
        caption_snippet="5 days in Kyoto",
        sender="instagram",
    )
    assert ok is True
    # Same link again -> deduped.
    dup = s.save_destination(
        platform="instagram",
        link="https://insta/p/ABC/",
        destination="Kyoto, Japan",
        confidence=0.9,
        source_field="caption",
        caption_snippet=None,
        sender="instagram",
    )
    assert dup is False
    rows = s.conn.execute("SELECT COUNT(*) c FROM destinations").fetchone()
    assert rows["c"] == 1


def test_save_multiple_destinations_for_same_link():
    s = make_store()
    ok1 = s.save_destination(
        platform="instagram",
        link="https://insta/p/MULTI/",
        destination="Tokyo, Japan",
        confidence=0.9,
        source_field="caption",
        caption_snippet="Tokyo then Kyoto",
        sender="instagram",
    )
    ok2 = s.save_destination(
        platform="instagram",
        link="https://insta/p/MULTI/",
        destination="Kyoto, Japan",
        confidence=0.8,
        source_field="caption",
        caption_snippet="Tokyo then Kyoto",
        sender="instagram",
    )
    assert ok1 is True
    assert ok2 is True
    rows = s.conn.execute(
        "SELECT COUNT(*) c FROM destinations WHERE link = 'https://insta/p/MULTI/'"
    ).fetchone()
    assert rows["c"] == 2


def test_save_multiple_landmarks_for_same_destination():
    s = make_store()
    ok1 = s.save_destination(
        platform="instagram",
        link="https://insta/p/ITIN/",
        destination="Madeira, Portugal",
        landmark="Pico do Arieiro",
        place_type="landmark",
        confidence=0.9,
        source_field="caption",
        caption_snippet="1-day itinerary",
        sender="instagram",
    )
    ok2 = s.save_destination(
        platform="instagram",
        link="https://insta/p/ITIN/",
        destination="Madeira, Portugal",
        landmark="Fanal Forest",
        place_type="landmark",
        confidence=0.9,
        source_field="caption",
        caption_snippet="1-day itinerary",
        sender="instagram",
    )
    # Re-saving the same destination+landmark pair still dedupes.
    dup = s.save_destination(
        platform="instagram",
        link="https://insta/p/ITIN/",
        destination="Madeira, Portugal",
        landmark="Pico do Arieiro",
        place_type="landmark",
        confidence=0.9,
        source_field="caption",
        caption_snippet=None,
        sender="instagram",
    )
    assert ok1 is True
    assert ok2 is True
    assert dup is False
    rows = s.conn.execute(
        "SELECT COUNT(*) c FROM destinations WHERE link = 'https://insta/p/ITIN/'"
    ).fetchone()
    assert rows["c"] == 2


def test_processed_tracking():
    s = make_store()
    assert s.is_processed("instagram", "msg1") is False
    s.mark_processed("instagram", "msg1")
    assert s.is_processed("instagram", "msg1") is True
    # idempotent
    s.mark_processed("instagram", "msg1")
    assert s.is_processed("instagram", "msg1") is True


def test_migrates_legacy_single_column_unique_link(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE destinations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            platform        TEXT NOT NULL,
            link            TEXT NOT NULL UNIQUE,
            destination     TEXT NOT NULL,
            confidence      REAL,
            source_field    TEXT,
            caption_snippet TEXT,
            sender          TEXT,
            created_at      TEXT NOT NULL
        );
        INSERT INTO destinations
            (platform, link, destination, confidence, source_field, caption_snippet, sender, created_at)
        VALUES ('instagram', 'https://insta/p/OLD/', 'Lisbon, Portugal', 0.9, 'caption', NULL, 'instagram', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    s = Store(db_path)
    # Old row preserved.
    row = s.conn.execute("SELECT * FROM destinations WHERE link = 'https://insta/p/OLD/'").fetchone()
    assert row["destination"] == "Lisbon, Portugal"
    # New constraint allows a second destination for the same link.
    ok = s.save_destination(
        platform="instagram",
        link="https://insta/p/OLD/",
        destination="Porto, Portugal",
        confidence=0.7,
        source_field="caption",
        caption_snippet=None,
        sender="instagram",
    )
    assert ok is True


def test_pending_lifecycle():
    s = make_store()
    s.add_pending(
        platform="instagram",
        thread_id="t1",
        link="https://insta/p/XYZ/",
        caption_snippet="somewhere pretty",
    )
    row = s.get_pending("instagram", "t1")
    assert row is not None
    assert row["link"] == "https://insta/p/XYZ/"
    s.clear_pending("instagram", "t1")
    assert s.get_pending("instagram", "t1") is None


def test_retry_queue_lifecycle():
    s = make_store()
    s.enqueue_retry("instagram", "msg1", '{"link": "x"}', retry_after=-1.0)
    due = s.due_retries()
    assert len(due) == 1
    row = due[0]
    assert row["platform"] == "instagram"
    assert row["item_id"] == "msg1"
    assert row["attempts"] == 0

    s.reschedule_retry(row["id"], retry_after=3600.0)
    assert s.due_retries() == []

    s.enqueue_retry("instagram", "msg2", '{"link": "y"}', retry_after=-1.0)
    due = s.due_retries()
    assert len(due) == 1
    s.delete_retry(due[0]["id"])
    assert s.due_retries() == []
