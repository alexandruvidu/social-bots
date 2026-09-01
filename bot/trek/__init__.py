"""Public entry point for pushing a destination into TREK.

bot/run.py imports push_destination and nothing else from this package —
it doesn't need to know MCP URLs, tokens, or transport details.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .client import call_tool as _call_tool
from .logic import push_place

if TYPE_CHECKING:
    from ..config import Config

__all__ = ["push_destination"]


def push_destination(
    cfg: "Config",
    *,
    platform: str,
    link: str,
    destination: str,
    landmark: str | None,
    place_type: str | None,
    topic: str | None,
    caption_snippet: str | None,
) -> bool:
    mcp_url = cfg.trek_url.rstrip("/") + "/mcp"

    def call_tool(name: str, arguments: dict) -> dict:
        return _call_tool(mcp_url, cfg.trek_api_token, name, arguments)

    return push_place(
        call_tool,
        platform=platform,
        link=link,
        destination=destination,
        landmark=landmark,
        place_type=place_type,
        topic=topic,
        caption_snippet=caption_snippet,
    )
