"""Shared ReAct graph factory used by the main agent and subagents.

Centralizes the agent/tools/route wiring so cancellation, iteration and time
limits, tracing, and vision handling behave identically in both execution paths.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.agent_helpers import analyze_screenshot
from agent.shared import sanitize_aimessage
from agent.trace import TraceCollector, activate_trace, display_content, duration_ms

logger = logging.getLogger(__name__)


def build_react_graph(
    state_type: type,
    bound_llm: Any,
    tools: list[Any],
    *,
    max_iter: int,
    max_secs: float,
    system_prompt: Callable[[], str],
    model_name: str = "",
    backend: str = "",
    trace_prefix: str = "",
    trace_context: dict[str, Any] | None = None,
    vision_llm: Any = None,
    vision_model: str = "",
    vision_failure_message: str = "Vision call failed: %s",
    vision_failure_args: tuple[Any, ...] = (),
) -> Any:
    """Compile a ReAct graph: agent -> (tools -> agent)* -> END.

    trace_prefix distinguishes event streams ("subagent_" for subagents);
    trace_context is merged into every emitted event payload.
    """
    context = trace_context or {}

    async def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        cancel = state.get("cancel_event")
        if cancel and cancel.is_set():
            return {
                "messages": state.get("messages", []),
                "stopped_reason": "cancelled",
            }
        msgs = list(state.get("messages", []))
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=system_prompt())] + msgs
        msgs = [sanitize_aimessage(m) if isinstance(m, AIMessage) else m for m in msgs]
        trace: TraceCollector | None = state.get("trace")
        llm_call = state.get("llm_calls_made", 0) + 1

        if (
            vision_llm is not None
            and msgs
            and isinstance(msgs[-1], ToolMessage)
            and getattr(msgs[-1], "name", None) == "computer_screenshot"
        ):
            msgs = await analyze_screenshot(
                msgs,
                vision_llm,
                vision_model,
                trace,
                trace_context=context,
                log=logger,
                failure_message=vision_failure_message,
                failure_args=vision_failure_args,
            )

        start = time.perf_counter()
        if trace:
            trace.emit(
                f"{trace_prefix}llm_started",
                **context,
                llm_call=llm_call,
                model=model_name,
                backend=backend,
                message_count=len(msgs),
            )
        try:
            resp = await bound_llm.ainvoke(msgs)
            cancel = state.get("cancel_event")
            if cancel and cancel.is_set():
                return {
                    "messages": state.get("messages", []),
                    "stopped_reason": "cancelled",
                }
        except Exception as exc:
            if trace:
                trace.emit(
                    f"{trace_prefix}llm_failed",
                    **context,
                    llm_call=llm_call,
                    duration_ms=duration_ms(start),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise
        reasoning_text = (getattr(resp, "additional_kwargs", {}) or {}).get(
            "reasoning_content", ""
        )
        if reasoning_text and trace:
            trace.emit(f"{trace_prefix}reasoning", **context, content=reasoning_text)
        if trace:
            trace.emit(
                f"{trace_prefix}llm_completed",
                **context,
                llm_call=llm_call,
                duration_ms=duration_ms(start),
                content=display_content(resp.content),
                tool_calls=getattr(resp, "tool_calls", None) or [],
                usage=getattr(resp, "usage_metadata", None) or {},
            )
        # Detect forced stop: LLM wants more tool calls but limits prevent next iteration
        tc = state.get("tool_calls_made", 0)
        stopped_reason = ""
        if getattr(resp, "tool_calls", None):
            if tc + 1 >= max_iter:
                stopped_reason = "max_iterations"
            elif (
                max_secs > 0
                and state.get("started_at")
                and time.perf_counter() - state["started_at"] >= max_secs
            ):
                stopped_reason = "max_seconds"
        result: dict[str, Any] = {
            "messages": msgs + [resp],
            "tool_calls_made": tc,
            "llm_calls_made": llm_call,
        }
        if stopped_reason:
            result["stopped_reason"] = stopped_reason
        return result

    tool_node = ToolNode(tools) if tools else None

    async def tools_node(state: dict[str, Any]) -> dict[str, Any]:
        if tool_node is None:
            return {}
        cancel = state.get("cancel_event")
        if cancel and cancel.is_set():
            return {"messages": state.get("messages", [])}
        with activate_trace(state.get("trace")):
            out = await tool_node.ainvoke(state)
        new_msgs = out.get("messages", [])
        return {
            "messages": list(state.get("messages", [])) + new_msgs,
            "tool_calls_made": state.get("tool_calls_made", 0) + 1,
        }

    def route(state: dict[str, Any]) -> str:
        if state.get("stopped_reason") == "cancelled":
            return END
        if state.get("tool_calls_made", 0) >= max_iter:
            return END
        if max_secs > 0 and state.get("started_at"):
            if time.perf_counter() - state["started_at"] >= max_secs:
                return END
        msgs = state.get("messages", [])
        if not msgs:
            return END
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    g = StateGraph(state_type)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()
