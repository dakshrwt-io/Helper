"""Unit tests for the dashboard execution trace; no model or MCP server required."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agent.graph import AgentGraph
from agent.mcp_adapter import _make_tool
from agent.trace import TraceCollector, activate_trace


class _FakeLLM:
    async def ainvoke(self, messages):
        return AIMessage(
            content="final response",
            usage_metadata={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        )


def test_chat_returns_streamable_llm_trace_events() -> None:
    async def run() -> None:
        agent = AgentGraph()
        agent._cfg = {"agent": {"max_iterations": 3}}
        agent._llm_with_tools = _FakeLLM()
        agent._build_graph()
        queue: asyncio.Queue[dict] = asyncio.Queue()

        result = await agent.chat("hello", trace_queue=queue)
        events = result["trace"]

        assert result["text"] == "final response"
        assert [event["type"] for event in events] == [
            "turn_started",
            "llm_started",
            "llm_completed",
            "memory_persisted",
            "turn_completed",
        ]
        assert events[2]["content"] == "final response"
        assert events[2]["usage"] == {
            "input_tokens": 5,
            "output_tokens": 7,
            "total_tokens": 12,
        }
        assert queue.qsize() == len(events)

    asyncio.run(run())


def test_mcp_tool_trace_includes_arguments_and_output() -> None:
    class FakeManager:
        def tool_server(self, tool_name: str) -> str:
            return "fake-server"

        async def call_tool(self, tool_name: str, arguments: dict):
            return SimpleNamespace(
                content=[SimpleNamespace(text=f"ran {arguments['command']}")],
                isError=False,
            )

    async def run() -> None:
        tool = _make_tool(
            SimpleNamespace(
                name="run_command",
                description="Runs a command",
                inputSchema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            ),
            FakeManager(),
        )
        trace = TraceCollector()
        with activate_trace(trace):
            result = await tool.ainvoke({"command": "dir"})

        assert result == "ran dir"
        assert trace.events[0]["type"] == "tool_started"
        assert trace.events[0]["arguments"] == {"command": "dir"}
        assert trace.events[1]["type"] == "tool_completed"
        assert trace.events[1]["output"] == "ran dir"

    asyncio.run(run())
