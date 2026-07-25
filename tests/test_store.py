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


# ── I2: several asks can be outstanding in one thread ──────────────────────

def _add(s, link, ask_msg_id, thread_id="t1"):
    s.add_pending(platform="instagram", thread_id=thread_id, link=link,
                  caption_snippet=None, ask_msg_id=ask_msg_id)


def test_two_asks_in_one_thread_both_survive():
    """The old (platform, thread_id) key made ask #2 destroy ask #1's row."""
    s = make_store()
    _add(s, "https://insta/p/ONE/", "ask_1")
    _add(s, "https://insta/p/TWO/", "ask_2")

    assert s.get_pending_by_ask_msg("instagram", "ask_1")["link"] == "https://insta/p/ONE/"
    assert s.get_pending_by_ask_msg("instagram", "ask_2")["link"] == "https://insta/p/TWO/"


def test_get_pending_returns_most_recent_open_row():
    s = make_store()
    _add(s, "https://insta/p/ONE/", "ask_1")
    _add(s, "https://insta/p/TWO/", "ask_2")

    assert s.get_pending("instagram", "t1")["link"] == "https://insta/p/TWO/"


def test_clearing_one_ask_leaves_the_other_open():
    s = make_store()
    _add(s, "https://insta/p/ONE/", "ask_1")
    _add(s, "https://insta/p/TWO/", "ask_2")

    s.clear_pending_by_ask_msg("instagram", "ask_2")

    assert s.get_pending_by_ask_msg("instagram", "ask_2") is None
    assert s.get_pending("instagram", "t1")["link"] == "https://insta/p/ONE/"


def test_re_asking_with_the_same_ask_id_replaces_that_row_only():
    s = make_store()
    _add(s, "https://insta/p/ONE/", "ask_1")
    _add(s, "https://insta/p/TWO/", "ask_2")
    _add(s, "https://insta/p/ONE-REDO/", "ask_1")

    assert s.get_pending_by_ask_msg("instagram", "ask_1")["link"] == "https://insta/p/ONE-REDO/"
    assert s.get_pending_by_ask_msg("instagram", "ask_2")["link"] == "https://insta/p/TWO/"
    rows = s.conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
    assert rows == 2


def test_ask_less_rows_stay_one_per_thread():
    """Preserved behaviour: a row with no ask id is only matchable by thread."""
    s = make_store()
    _add(s, "https://insta/p/ONE/", None)
    _add(s, "https://insta/p/TWO/", None)

    rows = s.conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
    assert rows == 1
    assert s.get_pending("instagram", "t1")["link"] == "https://insta/p/TWO/"


def test_clear_pending_does_not_destroy_other_outstanding_asks():
    s = make_store()
    _add(s, "https://insta/p/ONE/", "ask_1")
    _add(s, "https://insta/p/NOASK/", None)

    s.clear_pending("instagram", "t1")

    assert s.get_pending_by_ask_msg("instagram", "ask_1")["link"] == "https://insta/p/ONE/"


def test_threads_do_not_see_each_others_pending_rows():
    s = make_store()
    _add(s, "https://insta/p/ONE/", "ask_1", thread_id="t1")
    _add(s, "https://insta/p/TWO/", "ask_2", thread_id="t2")

    assert s.get_pending("instagram", "t1")["link"] == "https://insta/p/ONE/"
    assert s.get_pending("instagram", "t2")["link"] == "https://insta/p/TWO/"


def test_migrates_legacy_pending_table_without_dropping_rows(tmp_path):
    """A DB written under the old (platform, thread_id) key must carry over."""
    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE pending (
            platform        TEXT NOT NULL,
            thread_id       TEXT NOT NULL,
            link            TEXT NOT NULL,
            caption_snippet TEXT,
            ask_msg_id      TEXT,
            created_at      TEXT NOT NULL,
            PRIMARY KEY (platform, thread_id)
        );
        CREATE UNIQUE INDEX idx_pending_ask ON pending (platform, ask_msg_id)
            WHERE ask_msg_id IS NOT NULL;
        INSERT INTO pending VALUES
            ('instagram', 'thread_a', 'https://insta/p/OLD/', 'snip',
             'old_ask', '2026-06-28T09:53:18+00:00');
        """
    )
    conn.commit()
    conn.close()

    s = Store(db)

    row = s.get_pending("instagram", "thread_a")
    assert row["link"] == "https://insta/p/OLD/"
    assert row["ask_msg_id"] == "old_ask"
    assert row["caption_snippet"] == "snip"
    assert s.get_pending_by_ask_msg("instagram", "old_ask")["link"] == "https://insta/p/OLD/"
    # And the re-key actually took effect.
    cols = {r[1] for r in s.conn.execute("PRAGMA table_info(pending)")}
    assert "id" in cols
    _add(s, "https://insta/p/NEW/", "new_ask", thread_id="thread_a")
    assert s.get_pending_by_ask_msg("instagram", "old_ask") is not None


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "twice.sqlite"
    s1 = Store(db)
    _add(s1, "https://insta/p/ONE/", "ask_1")
    s1.close()

    s2 = Store(db)
    assert s2.get_pending_by_ask_msg("instagram", "ask_1")["link"] == "https://insta/p/ONE/"
    rows = s2.conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
    assert rows == 1


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
