"""MCP server manager: spawn subprocess MCP servers, expose tools to LangGraph."""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from typing import Literal

import yaml
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import CallToolResult, ImageContent, Tool as MCPTool

ImageContent.model_fields["type"].annotation = Literal["image", "blob"]
ImageContent.model_rebuild(force=True)
CallToolResult.model_rebuild(force=True)

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
        self._tools: dict[str, str] = {}  # tool_name -> server_name
        self._tool_defs: dict[str, MCPTool] = {}
        self._loaded = False
        self._healthy: dict[str, bool] = {}
        self._health_task: asyncio.Task | None = None
        self._health_interval: float = 30.0

    def _load_config(self) -> dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._expand(raw)
        return raw

    @staticmethod
    def _expand(obj: Any) -> None:
        import os
        if isinstance(obj, str):
            pass  # string values don't need in-place expansion
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    obj[k] = os.path.expandvars(v)  # type: ignore[arg-type]
                else:
                    MCPManager._expand(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    obj[i] = os.path.expandvars(v)  # type: ignore[index]
                else:
                    MCPManager._expand(v)

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
                self._healthy[name] = True

                tools_resp = await session.list_tools()
                for t in tools_resp.tools:
                    self._tools[t.name] = name
                    self._tool_defs[t.name] = t
                logger.info(
                    "MCP server '%s' started, %d tools: %s",
                    name,
                    len(tools_resp.tools),
                    [t.name for t in tools_resp.tools],
                )
            except Exception as e:
                logger.error("Failed to start MCP server '%s': %s", name, e)
                self._healthy[name] = False

        self._start_health_monitor()
        self._loaded = True

    async def list_tools_async(self) -> list[MCPTool]:
        """Fetch current tool list from all live sessions."""
        out: list[MCPTool] = []
        for name, session in self._sessions.items():
            try:
                resp = await session.list_tools()
                out.extend(resp.tools)
                for t in resp.tools:
                    self._tools[t.name] = name
                    self._tool_defs[t.name] = t
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
        if not self._healthy.get(server_name, False):
            raise RuntimeError(
                f"MCP server '{server_name}' is unhealthy. "
                f"Tool '{tool_name}' is unavailable."
            )
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments=arguments or {})
        return result

    # ── health monitor ────────────────────────────────────────────

    def _start_health_monitor(self) -> None:
        if self._health_task and not self._health_task.done():
            return
        self._health_task = asyncio.create_task(self._health_loop())

    def _stop_health_monitor(self) -> None:
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self._health_interval)
            for name in list(self._sessions.keys()):
                await self._ping_server(name)

    async def _ping_server(self, name: str) -> None:
        session = self._sessions.get(name)
        if session is None:
            self._healthy[name] = False
            return
        try:
            await asyncio.wait_for(session.list_tools(), timeout=10.0)
            if not self._healthy.get(name):
                logger.info("MCP server '%s' recovered — marked healthy", name)
            self._healthy[name] = True
        except Exception as exc:
            was_healthy = self._healthy.get(name, True)
            self._healthy[name] = False
            level = logging.WARNING if was_healthy else logging.DEBUG
            logger.log(level, "MCP server '%s' health check failed: %s", name, exc)

    @property
    def healthy_servers(self) -> list[str]:
        return [n for n, h in self._healthy.items() if h]

    async def stop(self) -> None:
        """Shut down all MCP server subprocesses."""
        self._stop_health_monitor()
        if self._exit_stack:
            await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        self._tool_defs.clear()
        self._healthy.clear()
        self._loaded = False
        logger.info("MCP manager stopped")

    @property
    def server_names(self) -> list[str]:
        return list(self._sessions.keys())

    @property
    def tool_names(self) -> list[str]:
        return [
            t for t, s in self._tools.items()
            if self._healthy.get(s, False)
        ]

    @property
    def tool_definitions(self) -> list[MCPTool]:
        return [
            tool for name, tool in self._tool_defs.items()
            if self._healthy.get(self._tools.get(name, ""), False)
        ]
