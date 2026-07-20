"""Regression tests for helpers shared by agent and subagent execution."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import agent.agent_helpers as helpers
from agent.trace import TraceCollector


class _VisionLLM:
    def __init__(self, response: SimpleNamespace | Exception) -> None:
        self.response = response
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _injected_screenshot(messages):
    return messages + [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
                {"type": "text", "text": "screenshot output"},
            ]
        )
    ]


@pytest.mark.asyncio
async def test_analyze_screenshot_keeps_main_agent_trace_shape(monkeypatch) -> None:
    monkeypatch.setattr(helpers, "inject_screenshots", _injected_screenshot)
    llm = _VisionLLM(
        SimpleNamespace(content="screen text", usage_metadata={"total_tokens": 7})
    )
    trace = TraceCollector()

    result = await helpers.analyze_screenshot(
        [HumanMessage(content="before screenshot")],
        llm,
        "vision-test",
        trace,
        log=logging.getLogger("agent.graph"),
        failure_message="Vision call failed: %s",
    )

    assert len(llm.messages) == 2
    assert isinstance(llm.messages[0], SystemMessage)
    assert result[0].content == "before screenshot"
    assert result[1].content == "screenshot output"
    assert result[2].content == "[Screen analysis]\nscreen text"
    assert [event["type"] for event in trace.events] == [
        "vision_started",
        "vision_completed",
    ]
    assert trace.events[0]["model"] == "vision-test"
    assert trace.events[0]["backend"] == "vision"
    assert trace.events[1]["content"] == "screen text"
    assert trace.events[1]["usage"] == {"total_tokens": 7}


@pytest.mark.asyncio
async def test_analyze_screenshot_keeps_subagent_failure_fallback(monkeypatch, caplog) -> None:
    monkeypatch.setattr(helpers, "inject_screenshots", _injected_screenshot)
    llm = _VisionLLM(RuntimeError("vision unavailable"))
    trace = TraceCollector()

    with caplog.at_level(logging.WARNING, logger="agent.subagents.manager"):
        result = await helpers.analyze_screenshot(
            [HumanMessage(content="before screenshot")],
            llm,
            "vision-test",
            trace,
            trace_context={"subagent_type": "browser"},
            log=logging.getLogger("agent.subagents.manager"),
            failure_message="Subagent '%s' vision call failed: %s",
            failure_args=("browser",),
        )

    assert [message.content for message in result] == [
        "before screenshot",
        "screenshot output",
    ]
    assert [event["type"] for event in trace.events] == [
        "vision_started",
        "vision_failed",
    ]
    assert trace.events[0]["subagent_type"] == "browser"
    assert trace.events[1]["subagent_type"] == "browser"
    assert trace.events[1]["error_type"] == "RuntimeError"
    assert "Subagent 'browser' vision call failed: vision unavailable" in caplog.text


def test_extract_final_ai_text_matches_tool_call_fallback() -> None:
    tool_call = AIMessage(
        content="I will use a tool",
        tool_calls=[{"name": "lookup", "args": {}, "id": "call-1", "type": "tool_call"}],
    )

    assert helpers.extract_final_ai_text([tool_call]) == "I will use a tool"
    assert helpers.extract_final_ai_text([tool_call, AIMessage(content="done")]) == "done"
