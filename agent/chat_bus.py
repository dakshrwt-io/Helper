"""Session-scoped pub-sub event bus."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger("agent.bus")
Subscriber = Callable[[dict], Awaitable[None]]


class ChatBus:
    """Broadcasts trace events and chat messages to all subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[int, tuple[str, Subscriber]] = {}
        self._global_subscribers: dict[int, Subscriber] = {}
        self._next_id = 0
        self._lock = asyncio.Lock()

    async def subscribe(self, session_id: str, callback: Subscriber) -> int:
        async with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers[sub_id] = (session_id, callback)
            return sub_id

    async def subscribe_all(self, callback: Subscriber) -> int:
        async with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._global_subscribers[sub_id] = callback
            return sub_id

    async def unsubscribe(self, sub_id: int) -> None:
        async with self._lock:
            self._subscribers.pop(sub_id, None)
            self._global_subscribers.pop(sub_id, None)

    async def publish(self, session_id: str, event: dict) -> None:
        async with self._lock:
            session_subs = [
                callback
                for subscribed_session, callback in self._subscribers.values()
                if subscribed_session == session_id
            ]
            global_subs = list(self._global_subscribers.values())
        all_subs = session_subs + global_subs
        if not all_subs:
            return
        event_with_session = {**event, "session_id": session_id}
        results = await asyncio.gather(
            *(sub(event_with_session) for sub in all_subs), return_exceptions=True
        )
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Bus subscriber error: %s", r)
