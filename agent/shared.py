"""Shared singleton for AgentGraph — used by both web and telegram providers."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.graph import AgentGraph

from agent.chat_bus import ChatBus

_graph: AgentGraph | None = None
_chat_lock: asyncio.Lock = asyncio.Lock()
_chat_bus: ChatBus = ChatBus()


def set_graph(graph: AgentGraph) -> None:
    global _graph
    _graph = graph


def get_graph() -> AgentGraph | None:
    return _graph


def get_chat_lock() -> asyncio.Lock:
    return _chat_lock


def get_chat_bus() -> ChatBus:
    return _chat_bus
