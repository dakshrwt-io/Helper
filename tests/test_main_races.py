"""Regression tests for WebSocket chat task races."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

import agent.main as main_module
from agent.shared import get_graph, set_graph


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


@asynccontextmanager
async def _unlocked_chat(_session_id: str):
    yield


def test_run_chat_with_trace_forwards_events_before_the_answer(monkeypatch) -> None:
    class FakeGraph:
        async def chat(self, _text, session_id, trace_queue, cancel_event=None):
            await trace_queue.put({"sequence": 1, "session_id": session_id})
            await trace_queue.put({"sequence": 2, "session_id": session_id})
            return {"text": "done"}

    async def run() -> None:
        ws = _FakeWebSocket()
        monkeypatch.setattr(main_module, "get_graph", lambda: FakeGraph())
        monkeypatch.setattr(main_module, "get_chat_lock", _unlocked_chat)

        result = await main_module._run_chat_with_trace(ws, "hello", "web_test")

        assert result == {"text": "done"}
        assert ws.sent == [
            {"type": "trace", "event": {"sequence": 1, "session_id": "web_test"}},
            {"type": "trace", "event": {"sequence": 2, "session_id": "web_test"}},
        ]

    asyncio.run(run())


def test_run_chat_with_trace_cancels_the_active_chat_task(monkeypatch) -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def chat(self, _text, session_id, trace_queue, cancel_event=None):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def run() -> None:
        graph = FakeGraph()
        ws = _FakeWebSocket()
        cancel_event = asyncio.Event()
        monkeypatch.setattr(main_module, "get_graph", lambda: graph)
        monkeypatch.setattr(main_module, "get_chat_lock", _unlocked_chat)

        task = asyncio.create_task(
            main_module._run_chat_with_trace(
                ws,
                "hello",
                "web_test",
                cancel_event,
            )
        )
        await graph.started.wait()
        cancel_event.set()

        with pytest.raises(asyncio.CancelledError, match="cancel requested"):
            await task

        assert graph.cancelled.is_set()
        assert ws.sent == []

    asyncio.run(run())


def test_websocket_cancel_sends_cancelled_before_the_chat_stops(monkeypatch) -> None:
    monkeypatch.setenv("WEB_TOKEN", "test-token")
    monkeypatch.setenv("WEB_COOKIE_SECURE", "false")
    original_graph = get_graph()

    class FakeGraph:
        mcp = None
        _chatdb = None

        async def chat(self, _text, session_id, trace_queue, cancel_event=None):
            await asyncio.Event().wait()

    set_graph(FakeGraph())
    try:
        with TestClient(main_module.app) as client:
            assert client.post("/auth/login", json={"token": "test-token"}).status_code == 200
            with client.websocket_connect(
                "/chat", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_json({"text": "hello"})
                assert ws.receive_json() == {"type": "thinking"}
                ws.send_json({"type": "cancel"})
                assert ws.receive_json() == {"type": "cancelled"}
    finally:
        set_graph(original_graph)
