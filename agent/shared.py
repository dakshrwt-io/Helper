"""Shared singleton and utilities for AgentGraph — used by both web and telegram providers."""
from __future__ import annotations

import asyncio
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
