"""Pure sync/push logic for getting social_bots destinations into TREK.

Talks to TREK only through an injected `call_tool(name, arguments)`
callable — see bot/trek/client.py for the real MCP transport, and
bot/trek/__init__.py for the thin wrapper that supplies it from config.
Keeping this module free of network code is what makes it testable with a
plain fake instead of mocking the MCP SDK's async internals.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("trek")

CallTool = Callable[[str, dict[str, Any]], dict[str, Any]]

MAX_LINKS = 30

_CATEGORY_NAME_BY_PLACE_TYPE = {
    "hotel": "Hotel",
    "restaurant": "Restaurant",
    "landmark": "Attraction",
}


def normalize_name(name: str) -> str:
    return " ".join(name.strip().split()).casefold()


def country_city_name(destination: str) -> str:
    """'Kyoto, Japan' -> 'Japan - Kyoto'; 'Japan' -> 'Japan' (no comma to split)."""
    parts = [p.strip() for p in destination.rsplit(",", 1)]
    if len(parts) == 2:
        city, country = parts
        return f"{country} - {city}"
    return destination.strip()


def build_note(
    landmark: str | None, place_type: str | None, caption_snippet: str | None
) -> str | None:
    parts: list[str] = []
    if landmark:
        suffix = f" ({place_type})" if place_type else ""
        parts.append(f"Landmark: {landmark}{suffix}")
    if caption_snippet:
        parts.append(caption_snippet.strip())
    return " — ".join(parts) if parts else None


def merge_links(
    existing_links: list[dict[str, str]], label: str, link: str
) -> tuple[list[dict[str, str]], bool]:
    if any(item["url"] == link for item in existing_links):
        return existing_links, False
    if len(existing_links) >= MAX_LINKS:
        log.warning("Not adding link %s: already has %d links (max)", link, MAX_LINKS)
        return existing_links, False
    return [*existing_links, {"label": label, "url": link}], True


def merge_notes(existing_notes: str | None, note_line: str | None) -> str | None:
    if not note_line:
        return existing_notes
    if not existing_notes:
        return note_line
    if note_line in existing_notes:
        return existing_notes
    return f"{existing_notes}\n{note_line}"


def _find_or_create_collection(call_tool: CallTool, name: str) -> int:
    existing = call_tool("list_collections", {})["collections"]
    for c in existing:
        if c["name"] == name:
            return c["id"]
    created = call_tool("create_collection", {"name": name})["collection"]
    return created["id"]


def _resolve_category_id(call_tool: CallTool, place_type: str | None) -> int | None:
    category_name = _CATEGORY_NAME_BY_PLACE_TYPE.get(place_type or "")
    if category_name is None:
        return None
    categories = call_tool("list_categories", {})["categories"]
    for c in categories:
        if c["name"] == category_name:
            return c["id"]
    return None


def _merge_link_into_place(
    call_tool: CallTool,
    place: dict[str, Any],
    label: str,
    link: str,
    note_line: str | None,
) -> bool:
    new_links, added = merge_links(place.get("links") or [], label, link)
    if not added:
        return False
    call_tool(
        "update_collection_place",
        {
            "placeId": place["id"],
            "links": new_links,
            "notes": merge_notes(place.get("notes"), note_line),
        },
    )
    place["links"] = new_links
    return True


def _merge_link_into_collection(
    call_tool: CallTool,
    collection_id: int,
    collection_links: list[dict[str, str]],
    label: str,
    link: str,
) -> bool:
    new_links, added = merge_links(collection_links, label, link)
    if not added:
        return False
    call_tool("update_collection", {"collectionId": collection_id, "links": new_links})
    return True


def push_place(
    call_tool: CallTool,
    *,
    platform: str,
    link: str,
    destination: str,
    landmark: str | None,
    place_type: str | None,
    topic: str | None,
    caption_snippet: str | None,
) -> bool:
    """Push one extracted destination/landmark into TREK.

    Never raises: every failure (network, MCP error, no geocoding match) is
    caught and logged, returning False, so a TREK problem can never abort
    the caller's post-processing or suppress its DM reply.
    """
    try:
        label = (topic or platform)[:120]
        collection_name = country_city_name(destination)
        collection_id = _find_or_create_collection(call_tool, collection_name)
        detail = call_tool("get_collection", {"collectionId": collection_id})
        collection = detail["collection"]
        places = detail["places"]

        if not landmark:
            collection_links = collection.get("links") or []
            if link in {item["url"] for item in collection_links}:
                return True
            _merge_link_into_collection(call_tool, collection_id, collection_links, label, link)
            return True

        by_name = {normalize_name(p["name"]): p for p in places}
        key = normalize_name(landmark)
        note_line = build_note(landmark, place_type, caption_snippet)
        existing = by_name.get(key)
        if existing is not None:
            existing_links = existing.get("links") or []
            if link in {item["url"] for item in existing_links}:
                return True
            _merge_link_into_place(call_tool, existing, label, link, note_line)
            return True

        geocoded = call_tool("search_place", {"query": f"{landmark}, {destination}"})
        results = geocoded.get("places") or []
        if not results:
            log.warning("No geocoding match for %r; not pushed to TREK.", landmark)
            return False
        top = results[0]

        category_id = _resolve_category_id(call_tool, place_type)
        resp = call_tool(
            "save_place_to_collection",
            {
                "collection_id": collection_id,
                "name": landmark,
                "lat": top.get("lat"),
                "lng": top.get("lng"),
                "address": top.get("address"),
                "osm_id": top.get("osm_id"),
                "category_id": category_id,
                "notes": note_line,
                "links": [{"label": label, "url": link}],
                "status": "want",
            },
        )
        if resp.get("duplicate"):
            duplicate_of = resp["duplicateOf"]
            duplicate_links = duplicate_of.get("links") or []
            if link in {item["url"] for item in duplicate_links}:
                return True
            _merge_link_into_place(call_tool, duplicate_of, label, link, note_line)
        return True
    except Exception:  # noqa: BLE001
        log.exception("Failed to push %r (%s) to TREK", destination, link)
        return False
