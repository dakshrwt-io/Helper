"""Adapter: expose MCP tools as LangChain-compatible tools for LangGraph."""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.tools import StructuredTool
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, create_model

from agent.mcp_manager import MCPManager
from agent.trace import duration_ms, get_active_trace


def _schema_to_pydantic(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic model from JSON schema properties."""
    props: dict[str, Any] = {}
    required = set(schema.get("required", []))
    for field, spec in (schema.get("properties") or {}).items():
        py_type = str
        if spec.get("type") == "integer":
            py_type = int
        elif spec.get("type") == "number":
            py_type = float
        elif spec.get("type") == "boolean":
            py_type = bool
        elif spec.get("type") == "array":
            py_type = list
        elif spec.get("type") == "object":
            py_type = dict
        default = ... if field in required else None
        props[field] = (py_type | None if default is None else py_type, default)
    return create_model(f"{tool_name}_Args", **props)  # type: ignore[arg-type]


def _make_tool(mcp_tool: MCPTool, mgr: MCPManager) -> StructuredTool:
    name = mcp_tool.name
    desc = mcp_tool.description or name
    schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}
    args_model = _schema_to_pydantic(name, schema)

    async def _run(**kwargs: Any) -> str:
        trace = get_active_trace()
        start = time.perf_counter()
        clean_args = {k: v for k, v in kwargs.items() if v is not None}
        if trace:
            trace.emit(
                "tool_started",
                tool_name=name,
                server=mgr.tool_server(name),
                arguments=clean_args,
            )
        try:
            result = await mgr.call_tool(name, clean_args)
        except Exception as exc:
            if trace:
                trace.emit(
                    "tool_failed",
                    tool_name=name,
                    server=mgr.tool_server(name),
                    duration_ms=duration_ms(start),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise
        parts: list[str] = []
        for c in result.content:
            text = getattr(c, "text", None)
            if text:
                parts.append(text)
            else:
                parts.append(json.dumps(c.model_dump(), default=str))
        out = "\n".join(parts)
        if trace:
            trace.emit(
                "tool_completed",
                tool_name=name,
                server=mgr.tool_server(name),
                duration_ms=duration_ms(start),
                is_error=bool(result.isError),
                output=out,
            )
        if result.isError:
            return f"[ERROR] {out}"
        return out

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=desc,
        args_schema=args_model,
    )


async def build_langchain_tools(mgr: MCPManager) -> list[StructuredTool]:
    """Fetch all MCP tools and wrap as LangChain StructuredTools."""
    mcp_tools = list(getattr(mgr, "tool_definitions", []))
    if not mcp_tools:
        mcp_tools = await mgr.list_tools_async()
    return [_make_tool(t, mgr) for t in mcp_tools]
