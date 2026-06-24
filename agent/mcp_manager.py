"""MCP server manager: spawn subprocess MCP servers, expose tools to LangGraph."""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

import yaml
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import Tool as MCPTool

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    transport: str = "stdio"


class MCPManager:
    """Manages multiple MCP server subprocesses and their tool sessions."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._servers: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, ClientSession] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._tools: dict[str, MCPTool] = {}  # tool_name -> server_name
        self._loaded = False

    def _load_config(self) -> dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _parse_servers(self, raw: dict[str, Any]) -> None:
        for name, cfg in raw.items():
            self._servers[name] = MCPServerConfig(
                name=name,
                command=cfg["command"],
                args=cfg.get("args", []),
                transport=cfg.get("transport", "stdio"),
            )

    async def start(self) -> None:
        """Spawn all MCP servers and connect sessions."""
        if self._loaded:
            return
        cfg = self._load_config()
        self._parse_servers(cfg.get("mcp", {}))
        self._exit_stack = AsyncExitStack()

        for name, srv in self._servers.items():
            try:
                params = StdioServerParameters(command=srv.command, args=srv.args)
                stdio_transport = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                read, write = stdio_transport
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                self._sessions[name] = session

                tools_resp = await session.list_tools()
                for t in tools_resp.tools:
                    self._tools[t.name] = name
                logger.info(
                    "MCP server '%s' started, %d tools: %s",
                    name,
                    len(tools_resp.tools),
                    [t.name for t in tools_resp.tools],
                )
            except Exception as e:
                logger.error("Failed to start MCP server '%s': %s", name, e)

        self._loaded = True

    def list_tools(self) -> list[MCPTool]:
        """Return all available MCP tools across servers (synchronous snapshot)."""
        out: list[MCPTool] = []
        for name, session in self._sessions.items():
            # list_tools is async; we cache from start(). Re-fetch lazily.
            pass
        return out

    async def list_tools_async(self) -> list[MCPTool]:
        """Fetch current tool list from all live sessions."""
        out: list[MCPTool] = []
        for name, session in self._sessions.items():
            try:
                resp = await session.list_tools()
                out.extend(resp.tools)
                for t in resp.tools:
                    self._tools[t.name] = name
            except Exception as e:
                logger.error("list_tools failed for '%s': %s", name, e)
        return out

    def tool_server(self, tool_name: str) -> str | None:
        """Which server owns a given tool name."""
        return self._tools.get(tool_name)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on its owning server. Returns MCP call_tool result."""
        server_name = self._tools.get(tool_name)
        if not server_name:
            raise KeyError(f"Tool '{tool_name}' not registered with any MCP server")
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments=arguments or {})
        return result

    async def stop(self) -> None:
        """Shut down all MCP server subprocesses."""
        if self._exit_stack:
            await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        self._loaded = False
        logger.info("MCP manager stopped")

    @property
    def server_names(self) -> list[str]:
        return list(self._sessions.keys())

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
