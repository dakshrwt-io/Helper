"""Pub-sub event bus for cross-provider observability."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger("agent.bus")
Subscriber = Callable[[dict], Awaitable[None]]


class ChatBus:
    """Broadcasts trace events and chat messages to all subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[int, Subscriber] = {}
        self._next_id = 0
        self._lock = asyncio.Lock()

    async def subscribe(self, callback: Subscriber) -> int:
        async with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers[sub_id] = callback
            return sub_id

    async def unsubscribe(self, sub_id: int) -> None:
        async with self._lock:
            self._subscribers.pop(sub_id, None)

    async def publish(self, event: dict) -> None:
        async with self._lock:
            subs = list(self._subscribers.values())
        if not subs:
            return
        results = await asyncio.gather(
            *(sub(event) for sub in subs), return_exceptions=True
        )
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Bus subscriber error: %s", r)
