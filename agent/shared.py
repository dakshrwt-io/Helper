"""Shared singleton and utilities for AgentGraph — used by both web and telegram providers."""
from __future__ import annotations

import asyncio
import contextvars
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from agent.graph import AgentGraph

from agent.chat_bus import ChatBus

_STRIP_ADDITIONAL_KW = frozenset({"reasoning_content", "reasoning_details"})


def sanitize_aimessage(m: AIMessage) -> AIMessage:
    if not (
        m.additional_kwargs
        and _STRIP_ADDITIONAL_KW.intersection(m.additional_kwargs)
    ):
        return m
    ak = {
        k: v
        for k, v in m.additional_kwargs.items()
        if k not in _STRIP_ADDITIONAL_KW
    }
    return m.model_copy(update={"additional_kwargs": ak})

_graph: AgentGraph | None = None
_chat_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_chat_lock_users: dict[str, int] = defaultdict(int)
_chat_locks_guard = asyncio.Lock()
_chat_bus: ChatBus = ChatBus()
_active_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_session_id", default=None
)


def set_graph(graph: AgentGraph) -> None:
    global _graph
    _graph = graph


def get_graph() -> AgentGraph | None:
    return _graph


@asynccontextmanager
async def get_chat_lock(session_id: str):
    async with _chat_locks_guard:
        lock = _chat_locks[session_id]
        _chat_lock_users[session_id] += 1
    try:
        async with lock:
            yield
    finally:
        async with _chat_locks_guard:
            _chat_lock_users[session_id] -= 1
            if _chat_lock_users[session_id] == 0 and not lock.locked():
                _chat_lock_users.pop(session_id, None)
                _chat_locks.pop(session_id, None)


def get_active_session_id() -> str | None:
    return _active_session_id.get()


@contextmanager
def activate_session(session_id: str):
    token = _active_session_id.set(session_id)
    try:
        yield
    finally:
        _active_session_id.reset(token)


def get_chat_bus() -> ChatBus:
    return _chat_bus
