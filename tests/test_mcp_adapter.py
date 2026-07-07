from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent.mcp_adapter import build_langchain_tools


def test_build_langchain_tools_uses_cached_definitions() -> None:
    class CachedManager:
        @property
        def tool_definitions(self):
            return [
                SimpleNamespace(
                    name="read_file",
                    description="Read a file",
                    inputSchema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                )
            ]

        async def list_tools_async(self):
            raise AssertionError("cached tool definitions should be used")

        def tool_server(self, tool_name: str) -> str:
            return "filesystem"

    async def run() -> None:
        tools = await build_langchain_tools(CachedManager())

        assert [tool.name for tool in tools] == ["read_file"]

    asyncio.run(run())
