"""Internal helpers shared by main-agent and subagent execution paths."""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.tools.computer import inject_screenshots, strip_image_content
from agent.trace import TraceCollector, display_content, duration_ms


_SCREEN_READER_PROMPT = (
    "You are a screen reader. Describe this screenshot in exact detail.\n"
    "Include:\n"
    "- Every visible window (title, position, content)\n"
    "- All UI elements (buttons, fields, menus, icons)\n"
    "- Any visible text (read it verbatim)\n"
    "- Mouse cursor position if visible\n"
    "- Coordinate grid labels (red numbers at edges)\n"
    "- Taskbar / system tray state\n"
    "\n"
    "Be specific and thorough. Your description will be used by another AI "
    "to decide what actions to take on this screen."
)


def _trace_fields(
    context: dict[str, Any] | None,
    **fields: Any,
) -> dict[str, Any]:
    """Add optional context before the standard vision trace fields."""
    return {**(context or {}), **fields}


async def analyze_screenshot(
    messages: list[Any],
    vision_llm: Any,
    vision_model: str,
    trace: TraceCollector | None,
    *,
    trace_context: dict[str, Any] | None = None,
    log: logging.Logger,
    failure_message: str,
    failure_args: tuple[Any, ...] = (),
) -> list[Any]:
    """Describe the latest screenshot and return text-only acting messages.

    Callers decide when a screenshot is present. This helper deliberately keeps
    the existing best-effort behavior: a failed vision call is traced and image
    blocks are still removed so a text-only acting model can continue.
    """
    messages = inject_screenshots(messages)
    vision_messages = [
        SystemMessage(content=_SCREEN_READER_PROMPT),
        messages[-1],
    ]
    vision_start = time.perf_counter()
    if trace:
        trace.emit(
            "vision_started",
            **_trace_fields(
                trace_context,
                model=vision_model,
                backend="vision",
            ),
        )

    try:
        vision_response = await vision_llm.ainvoke(vision_messages)
        vision_text = display_content(vision_response.content)
        messages.append(HumanMessage(content=f"[Screen analysis]\n{vision_text}"))
        messages = strip_image_content(messages)
        if trace:
            trace.emit(
                "vision_completed",
                **_trace_fields(
                    trace_context,
                    duration_ms=duration_ms(vision_start),
                    content=vision_text,
                    usage=getattr(vision_response, "usage_metadata", None) or {},
                ),
            )
    except Exception as exc:
        log.warning(failure_message, *failure_args, exc)
        messages = strip_image_content(messages)
        if trace:
            trace.emit(
                "vision_failed",
                **_trace_fields(
                    trace_context,
                    duration_ms=duration_ms(vision_start),
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            )
    return messages


def extract_final_ai_text(messages: list[Any]) -> str:
    """Return the last final AI text, falling back to any AI response."""
    final = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            final = display_content(message.content)
            break
    if final:
        return final

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return display_content(message.content) if message.content else ""
    return ""
