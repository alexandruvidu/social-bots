"""Offline tests for bot.trek.logic — no network calls.

A live, opt-in check of the real MCP transport lives in this same file,
gated by RUN_LIVE_TESTS=1 (see the bottom of this file, added in Task 3),
matching this project's existing convention (tests/test_extract.py).
"""
from bot.trek.logic import (
    build_note,
    country_city_name,
    merge_links,
    merge_notes,
    normalize_name,
    push_place,
)


def test_normalize_name_trims_and_casefolds():
    assert normalize_name("  Fushimi Inari  ") == "fushimi inari"


def test_country_city_name_splits_city_country():
    assert country_city_name("Kyoto, Japan") == "Japan - Kyoto"


def test_country_city_name_bare_country():
    assert country_city_name("Japan") == "Japan"


def test_build_note_all_fields():
    assert build_note("Fushimi Inari", "landmark", "wow") == "Landmark: Fushimi Inari (landmark) — wow"


def test_merge_links_dedup_and_cap():
    links, added = merge_links([], "instagram", "https://x/1")
    assert added and links == [{"label": "instagram", "url": "https://x/1"}]
    links2, added2 = merge_links(links, "instagram", "https://x/1")
    assert not added2 and links2 is links


def test_merge_notes_append():
    assert merge_notes("a", "b") == "a\nb"


class FakeTrek:
    """In-memory stand-in for the TREK MCP tools push_place calls."""

    CATEGORIES = [
        {"id": 1, "name": "Hotel"},
        {"id": 2, "name": "Restaurant"},
        {"id": 3, "name": "Attraction"},
        {"id": 10, "name": "Other"},
    ]

    def __init__(self):
        self.collections: list[dict] = []
        self.places: dict[int, list[dict]] = {}
        self._next_cid = 1
        self._next_pid = 1
        self.geocode: dict[str, tuple[float, float]] = {}
        self.calls: list[tuple[str, dict]] = []

    def seed_collection(self, name: str, links=None) -> int:
        cid = self._next_cid
        self._next_cid += 1
        self.collections.append({"id": cid, "name": name, "links": links or []})
        self.places[cid] = []
        return cid

    def seed_place(self, collection_id: int, **fields) -> dict:
        place = {
            "id": self._next_pid,
            "collection_id": collection_id,
            "name": fields.get("name", ""),
            "lat": fields.get("lat"),
            "lng": fields.get("lng"),
            "notes": fields.get("notes"),
            "links": fields.get("links", []),
            "status": fields.get("status", "idea"),
            "category_id": fields.get("category_id"),
        }
        self._next_pid += 1
        self.places[collection_id].append(place)
        return place

    def __call__(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            raise AssertionError(f"unexpected tool call: {name}")
        return method(arguments)

    def _tool_list_collections(self, args):
        return {"collections": list(self.collections), "incomingInvites": []}

    def _tool_create_collection(self, args):
        cid = self.seed_collection(args["name"])
        return {"collection": {"id": cid, "name": args["name"], "links": []}}

    def _tool_get_collection(self, args):
        cid = args["collectionId"]
        return {
            "collection": dict(next(c for c in self.collections if c["id"] == cid)),
            "places": [dict(p) for p in self.places[cid]],
        }

    def _tool_update_collection(self, args):
        cid = args["collectionId"]
        col = next(c for c in self.collections if c["id"] == cid)
        if "links" in args:
            col["links"] = args["links"]
        return {"collection": dict(col)}

    def _tool_list_categories(self, args):
        return {"categories": list(self.CATEGORIES)}

    def _tool_search_place(self, args):
        lat, lng = self.geocode.get(args["query"], (1.0, 2.0))
        return {
            "places": [
                {
                    "lat": lat,
                    "lng": lng,
                    "osm_id": "way:1",
                    "name": args["query"],
                    "address": f"addr for {args['query']}",
                }
            ],
            "source": "openstreetmap",
        }

    def _tool_save_place_to_collection(self, args):
        cid = args["collection_id"]
        for p in self.places[cid]:
            if p["name"].casefold() == args["name"].casefold():
                return {"duplicate": True, "duplicateOf": dict(p)}
        place = {
            "id": self._next_pid,
            "collection_id": cid,
            "name": args["name"],
            "lat": args.get("lat"),
            "lng": args.get("lng"),
            "address": args.get("address"),
            "osm_id": args.get("osm_id"),
            "category_id": args.get("category_id"),
            "notes": args.get("notes"),
            "links": args.get("links") or [],
            "status": args.get("status", "idea"),
        }
        self._next_pid += 1
        self.places[cid].append(place)
        return {"place": dict(place)}

    def _tool_update_collection_place(self, args):
        pid = args["placeId"]
        for plist in self.places.values():
            for p in plist:
                if p["id"] == pid:
                    for key in ("name", "notes", "lat", "lng", "status", "links"):
                        if key in args:
                            p[key] = args[key]
                    return {"place": dict(p)}
        raise AssertionError(f"unknown placeId {pid}")


def _push(trek, **overrides):
    kwargs = dict(
        platform="instagram",
        link="https://instagram.com/p/1",
        destination="Kyoto, Japan",
        landmark=None,
        place_type=None,
        topic=None,
        caption_snippet=None,
    )
    kwargs.update(overrides)
    return push_place(trek, **kwargs)


def test_push_creates_collection_and_landmark_place_with_category():
    trek = FakeTrek()
    trek.geocode["Fushimi Inari, Kyoto, Japan"] = (35.0, 135.7)

    ok = _push(
        trek,
        landmark="Fushimi Inari",
        place_type="landmark",
        topic="rules about trains in Japan",
        caption_snippet="wow",
    )

    assert ok is True
    assert trek.collections == [{"id": 1, "name": "Japan - Kyoto", "links": []}]
    [place] = trek.places[1]
    assert place["name"] == "Fushimi Inari"
    assert place["lat"] == 35.0 and place["lng"] == 135.7
    assert place["address"] == "addr for Fushimi Inari, Kyoto, Japan"
    assert place["osm_id"] == "way:1"
    assert place["category_id"] == 3  # Attraction
    assert place["links"] == [{"label": "rules about trains in Japan", "url": "https://instagram.com/p/1"}]
    assert place["status"] == "want"
    assert place["notes"] == "Landmark: Fushimi Inari (landmark) — wow"


def test_push_falls_back_to_platform_label_when_no_topic():
    trek = FakeTrek()
    _push(trek, landmark="Fushimi Inari", place_type="landmark", topic=None)
    [place] = trek.places[1]
    assert place["links"][0]["label"] == "instagram"


def test_push_hotel_category():
    trek = FakeTrek()
    _push(trek, landmark="Park Hyatt", place_type="hotel")
    [place] = trek.places[1]
    assert place["category_id"] == 1


def test_push_restaurant_category():
    trek = FakeTrek()
    _push(trek, landmark="Ichiran", place_type="restaurant")
    [place] = trek.places[1]
    assert place["category_id"] == 2


def test_push_no_landmark_goes_on_collection_links_not_a_place():
    trek = FakeTrek()

    ok = _push(trek, landmark=None, topic="general vibes of Kyoto")

    assert ok is True
    assert trek.places[1] == []
    col = trek.collections[0]
    assert col["links"] == [{"label": "general vibes of Kyoto", "url": "https://instagram.com/p/1"}]


def test_push_no_landmark_skips_when_link_already_on_collection():
    trek = FakeTrek()
    trek.seed_collection("Japan - Kyoto", links=[{"label": "x", "url": "https://instagram.com/p/1"}])

    _push(trek, landmark=None)

    assert [name for name, _ in trek.calls if name == "update_collection"] == []


def test_push_merges_into_existing_place_by_name():
    trek = FakeTrek()
    cid = trek.seed_collection("Japan - Kyoto")
    trek.seed_place(cid, name="fushimi inari", links=[{"label": "tiktok", "url": "https://tiktok.com/1"}], notes="old")

    ok = _push(trek, landmark="Fushimi Inari", place_type="landmark", link="https://instagram.com/p/2", caption_snippet="wow")

    assert ok is True
    assert [name for name, _ in trek.calls if name == "save_place_to_collection"] == []
    [place] = trek.places[cid]
    assert place["links"] == [
        {"label": "tiktok", "url": "https://tiktok.com/1"},
        {"label": "instagram", "url": "https://instagram.com/p/2"},
    ]
    assert place["notes"] == "old\nLandmark: Fushimi Inari (landmark) — wow"


def test_push_skips_when_link_already_synced_on_a_place():
    trek = FakeTrek()
    cid = trek.seed_collection("Japan - Kyoto")
    trek.seed_place(cid, name="Fushimi Inari", links=[{"label": "instagram", "url": "https://instagram.com/p/1"}])

    _push(trek, landmark="Fushimi Inari")

    assert all(name not in ("update_collection_place", "save_place_to_collection") for name, _ in trek.calls)


def test_push_multiple_landmarks_same_link_both_get_created():
    """Regression test: two different landmarks extracted from the same post
    share the same `link` (bot/run.py's _save_places calls push_place once per
    landmark with the one post URL). Both must end up as their own places
    with the link attached, not have the second landmark silently dropped
    because the link already appears elsewhere in the collection.
    """
    trek = FakeTrek()

    ok1 = _push(trek, landmark="Fushimi Inari", place_type="landmark")
    ok2 = _push(trek, landmark="Kiyomizu-dera", place_type="landmark")

    assert ok1 is True and ok2 is True
    [cid] = [c["id"] for c in trek.collections]
    places_by_name = {p["name"]: p for p in trek.places[cid]}
    assert set(places_by_name) == {"Fushimi Inari", "Kiyomizu-dera"}
    for place in places_by_name.values():
        assert place["links"] == [{"label": "instagram", "url": "https://instagram.com/p/1"}]


def test_push_merges_into_treks_reported_duplicate():
    class AlwaysDuplicateTrek(FakeTrek):
        def _tool_save_place_to_collection(self, args):
            existing = self.places[args["collection_id"]][0]
            return {"duplicate": True, "duplicateOf": dict(existing)}

    trek = AlwaysDuplicateTrek()
    cid = trek.seed_collection("Japan - Kyoto")
    trek.seed_place(cid, name="Fushimi Inari Shrine (old name)", links=[])

    ok = _push(trek, landmark="Fushimi Inari", place_type="landmark")

    assert ok is True
    [place] = trek.places[cid]
    assert place["links"] == [{"label": "instagram", "url": "https://instagram.com/p/1"}]


def test_push_returns_false_when_geocoding_finds_nothing():
    class NoResultsTrek(FakeTrek):
        def _tool_search_place(self, args):
            return {"places": [], "source": "openstreetmap"}

    trek = NoResultsTrek()
    ok = _push(trek, landmark="Fushimi Inari", place_type="landmark")
    assert ok is False


def test_push_never_raises_on_internal_error():
    def broken_call_tool(name, arguments):
        raise RuntimeError("network exploded")

    ok = push_place(
        broken_call_tool,
        platform="instagram",
        link="https://instagram.com/p/1",
        destination="Kyoto, Japan",
        landmark=None,
        place_type=None,
        topic=None,
        caption_snippet=None,
    )
    assert ok is False


import os

import pytest

from bot.trek.client import call_tool as _real_call_tool

pytestmark_live = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_TESTS") and os.environ.get("TREK_URL") and os.environ.get("TREK_API_TOKEN")),
    reason="live TREK test; set RUN_LIVE_TESTS=1, TREK_URL, and TREK_API_TOKEN to run",
)


@pytestmark_live
def test_live_list_collections_reaches_real_trek():
    mcp_url = os.environ["TREK_URL"].rstrip("/") + "/mcp"
    token = os.environ["TREK_API_TOKEN"]
    result = _real_call_tool(mcp_url, token, "list_collections", {})
    assert "collections" in result
    assert isinstance(result["collections"], list)


from types import SimpleNamespace

import bot.trek as trek_pkg


def test_push_destination_builds_mcp_url_and_delegates(monkeypatch):
    captured = {}

    def fake_push_place(call_tool, **kwargs):
        captured["kwargs"] = kwargs
        captured["tool_result"] = call_tool("some_tool", {"a": 1})
        return True

    def fake_client_call_tool(mcp_url, token, name, arguments):
        captured["mcp_url"] = mcp_url
        captured["token"] = token
        captured["name"] = name
        captured["arguments"] = arguments
        return {"ok": True}

    monkeypatch.setattr(trek_pkg, "push_place", fake_push_place)
    monkeypatch.setattr(trek_pkg, "_call_tool", fake_client_call_tool)

    cfg = SimpleNamespace(trek_url="http://trek.local:3000/", trek_api_token="trek_abc")

    ok = trek_pkg.push_destination(
        cfg,
        platform="instagram",
        link="L",
        destination="D",
        landmark=None,
        place_type=None,
        topic=None,
        caption_snippet=None,
    )

    assert ok is True
    assert captured["mcp_url"] == "http://trek.local:3000/mcp"  # trailing slash stripped
    assert captured["token"] == "trek_abc"
    assert captured["name"] == "some_tool"
    assert captured["kwargs"]["destination"] == "D"
    assert captured["tool_result"] == {"ok": True}
