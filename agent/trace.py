"""In-memory, per-turn execution tracing for the web dashboard.

Trace data is deliberately scoped to the active chat request. It is streamed to
the connected browser and is not written to the application logs or database.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


_active_trace: contextvars.ContextVar["TraceCollector | None"] = contextvars.ContextVar(
    "active_trace", default=None
)


def _json_safe(value: Any) -> Any:
    """Convert arbitrary tool and model values into WebSocket-safe JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    return str(value)


@dataclass
class TraceCollector:
    """Collect ordered events for one agent turn and optionally stream them."""

    queue: asyncio.Queue[dict[str, Any]] | None = None
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list[dict[str, Any]] = field(default_factory=list)
    _sequence: int = 0

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        self._sequence += 1
        event = {
            "type": event_type,
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn_id": self.turn_id,
            **_json_safe(payload),
        }
        self.events.append(event)
        if self.queue is not None:
            self.queue.put_nowait(event)
        return event


def get_active_trace() -> TraceCollector | None:
    return _active_trace.get()


@contextmanager
def activate_trace(trace: TraceCollector | None) -> Iterator[None]:
    """Make a trace available to nested asynchronous LangChain tool calls."""
    if trace is None:
        yield
        return
    token = _active_trace.set(trace)
    try:
        yield
    finally:
        _active_trace.reset(token)


def duration_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def display_content(content: Any) -> str:
    """Format an LLM response content field for the trace viewer."""
    if isinstance(content, str):
        return content
    return json.dumps(_json_safe(content), ensure_ascii=False, indent=2)
