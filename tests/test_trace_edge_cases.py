"""Unit tests for trace events on edge cases: cost cap block, tool failure, no-tools turn."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from agent.graph import AgentGraph
from agent.mcp_adapter import _make_tool
from agent.trace import TraceCollector, activate_trace


class _FakeLLM:
    async def ainvoke(self, messages):
        return AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )


def test_cost_cap_trace_emits_turn_blocked() -> None:
    async def run() -> None:
        agent = AgentGraph()
        agent._cfg = {"agent": {"max_iterations": 3}}
        agent._daily_cap = 1.0

        # fake chatdb that reports over-cap spend
        class FakeDB:
            def spent_today(self) -> float:
                return 1.5
            def get_history(self, *a, **k):
                return []
            def add_message(self, *a, **k):
                pass
            def add_cost(self, *a, **k):
                pass
        agent._chatdb = FakeDB()
        agent._vector = None

        queue: asyncio.Queue[dict] = asyncio.Queue()
        result = await agent.chat("hello", trace_queue=queue)
        events = result["trace"]
        types = [e["type"] for e in events]

        assert types == ["turn_started", "turn_blocked", "turn_completed"]
        assert events[1]["reason"] == "daily_cost_cap"
        assert "cap" in result["text"].lower() and "reached" in result["text"].lower()
        assert queue.qsize() == len(events)

    asyncio.run(run())


def test_tool_failure_trace_emits_tool_failed() -> None:
    class FailingManager:
        def tool_server(self, tool_name: str) -> str:
            return "fs"

        async def call_tool(self, tool_name: str, arguments: dict):
            raise RuntimeError("connection lost")

    async def run() -> None:
        tool = _make_tool(
            SimpleNamespace(
                name="read_file",
                description="Reads a file",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            FailingManager(),
        )
        trace = TraceCollector()
        with activate_trace(trace):
            with pytest.raises(RuntimeError, match="connection lost"):
                await tool.ainvoke({"path": "/tmp/none"})

        assert len(trace.events) == 2
        assert trace.events[0]["type"] == "tool_started"
        assert trace.events[0]["tool_name"] == "read_file"
        assert trace.events[1]["type"] == "tool_failed"
        assert trace.events[1]["error_type"] == "RuntimeError"
        assert "connection lost" in trace.events[1]["error"]

    asyncio.run(run())


def test_trace_queue_receives_all_events_in_order() -> None:
    async def run() -> None:
        agent = AgentGraph()
        agent._cfg = {"agent": {"max_iterations": 3}}
        agent._llm_with_tools = _FakeLLM()
        agent._build_graph()
        agent._chatdb = None
        agent._vector = None

        queue: asyncio.Queue[dict] = asyncio.Queue()
        await agent.chat("hi", trace_queue=queue)

        queued_types = []
        while not queue.empty():
            queued_types.append(queue.get_nowait()["type"])

        # all events should be in queue, in emission order
        assert queued_types == [
            "turn_started",
            "llm_started",
            "llm_completed",
            "memory_persisted",
            "turn_completed",
        ]

    asyncio.run(run())
