"""Real MCP transport to a TREK instance.

Kept separate from bot.trek.logic's pure logic so that module's tests can
inject a fake `call_tool` instead of mocking the MCP SDK's async internals.

`httpx2`/`mcp` are imported lazily inside `_call_tool_async` rather than at
module level: bot.run and bot.webhook import this module unconditionally
(via bot.trek), and TREK is optional (unset TREK_URL/TREK_API_TOKEN means
the push is skipped entirely) — a module-level import here would make the
whole bot fail to even start on an install that never configured TREK and
therefore never ran `pip install` for these two packages. This mirrors
bot/run.py's run_once(), which imports bot.sources.instagram lazily inside
the function for the identical "optional dependency" reason.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any


async def _call_tool_async(
    mcp_url: str, token: str, name: str, arguments: dict[str, Any]
) -> Any:
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {token}"}
    timeout = httpx2.Timeout(10.0, read=30.0)
    async with httpx2.AsyncClient(
        headers=headers, timeout=timeout, follow_redirects=True
    ) as http_client:
        async with streamable_http_client(
            mcp_url, http_client=http_client
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                if result.isError:
                    text = result.content[0].text if result.content else "unknown MCP error"
                    raise RuntimeError(f"MCP tool {name!r} failed: {text}")
                if not result.content:
                    return {}
                return json.loads(result.content[0].text)


def call_tool(mcp_url: str, token: str, name: str, arguments: dict[str, Any]) -> Any:
    """Synchronous wrapper — the rest of this service is plain sync code."""
    return asyncio.run(_call_tool_async(mcp_url, token, name, arguments))
