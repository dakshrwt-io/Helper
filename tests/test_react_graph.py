"""Coverage for the shared ReAct graph factory (agent + subagent paths)."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from typing_extensions import TypedDict

from agent.react import build_react_graph
from agent.trace import TraceCollector


class _State(TypedDict, total=False):
    messages: list
    tool_calls_made: int
    llm_calls_made: int
    trace: TraceCollector
    started_at: float
    stopped_reason: str
    cancel_event: object


class _CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, msgs):
        self.calls += 1
        return AIMessage(content="done")


def _build(llm, **kwargs):
    params = dict(
        max_iter=3,
        max_secs=0,
        system_prompt=lambda: "sys",
    )
    params.update(kwargs)
    return build_react_graph(_State, llm, [], **params)


@pytest.mark.asyncio
async def test_cancelled_event_short_circuits_before_llm_call() -> None:
    llm = _CountingLLM()
    graph = _build(llm)
    cancel = asyncio.Event()
    cancel.set()

    result = await graph.ainvoke(
        {"messages": [], "cancel_event": cancel, "started_at": 1.0}
    )

    assert result["stopped_reason"] == "cancelled"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_trace_event_names_follow_prefix() -> None:
    for prefix in ("", "subagent_"):
        trace = TraceCollector(queue=asyncio.Queue())
        graph = _build(_CountingLLM(), trace_prefix=prefix)

        result = await graph.ainvoke({"messages": [], "trace": trace})

        assert result["messages"][-1].content == "done"
        assert [e["type"] for e in trace.events] == [
            f"{prefix}llm_started",
            f"{prefix}llm_completed",
        ]


@pytest.mark.asyncio
async def test_system_prompt_injected_once() -> None:
    calls: list[list] = []

    class _CaptureLLM:
        async def ainvoke(self, msgs):
            calls.append(list(msgs))
            return AIMessage(content="done")

    graph = _build(_CaptureLLM(), system_prompt=lambda: "prompt-v1")

    await graph.ainvoke({"messages": []})

    assert calls[0][0].content == "prompt-v1"
    assert len(calls[0]) == 1  # only the system message; no duplicates on re-entry
