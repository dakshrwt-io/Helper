from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent.chat_bus import ChatBus
from agent.main import app
from agent.memory.db import ChatDB
from agent.mcp_manager import MCPManager, MCPServerConfig
from agent.shared import activate_session, get_chat_lock, get_graph, set_graph
from agent.tools.computer import (
    _authorize_action,
    _pop_screenshot,
    _push_screenshot,
    grant_desktop_lease,
    revoke_desktop_lease,
)


@pytest.mark.asyncio
async def test_chat_bus_only_delivers_matching_session() -> None:
    bus = ChatBus()
    received_a: list[dict] = []
    received_b: list[dict] = []

    async def receive_a(event: dict) -> None:
        received_a.append(event)

    async def receive_b(event: dict) -> None:
        received_b.append(event)

    await bus.subscribe("a", receive_a)
    await bus.subscribe("b", receive_b)
    await bus.publish("a", {"type": "answer"})

    assert received_a == [{"type": "answer", "session_id": "a"}]
    assert received_b == []


@pytest.mark.asyncio
async def test_chat_lock_serializes_only_matching_session() -> None:
    entered_a = asyncio.Event()
    release_a = asyncio.Event()
    entered_b = asyncio.Event()

    async def first_a() -> None:
        async with get_chat_lock("a"):
            entered_a.set()
            await release_a.wait()

    async def second_a() -> None:
        await entered_a.wait()
        async with get_chat_lock("a"):
            raise AssertionError("same session entered before release")

    async def session_b() -> None:
        await entered_a.wait()
        async with get_chat_lock("b"):
            entered_b.set()

    first = asyncio.create_task(first_a())
    blocked = asyncio.create_task(second_a())
    other = asyncio.create_task(session_b())
    await entered_b.wait()
    assert not blocked.done()
    blocked.cancel()
    release_a.set()
    await asyncio.gather(first, blocked, other, return_exceptions=True)


def test_websocket_requires_auth_and_owns_session_id(monkeypatch) -> None:
    monkeypatch.setenv("WEB_TOKEN", "test-token")
    monkeypatch.setenv("WEB_COOKIE_SECURE", "false")
    original_graph = get_graph()

    class FakeGraph:
        mcp = None
        _chatdb = None

        def __init__(self) -> None:
            self.session_id = ""

        async def chat(self, _text, session_id, trace_queue, cancel_event=None):
            self.session_id = session_id
            return {"text": "ok"}

    graph = FakeGraph()
    set_graph(graph)
    try:
        with TestClient(app) as client:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/chat", headers={"origin": "http://testserver"}
                ):
                    pass
            assert client.post("/auth/login", json={"token": "wrong"}).status_code == 401
            assert client.post("/auth/login", json={"token": "test-token"}).status_code == 200
            with client.websocket_connect(
                "/chat", headers={"origin": "http://testserver"}
            ) as ws:
                ws.send_json({"text": "hello", "session_id": "telegram_123"})
                assert ws.receive_json()["type"] == "thinking"
                assert ws.receive_json() == {"type": "answer", "text": "ok"}
        assert graph.session_id.startswith("web_")
        assert graph.session_id != "telegram_123"
    finally:
        set_graph(original_graph)


def test_export_pairs_rows_within_each_session() -> None:
    db = ChatDB(":memory:")
    db.add_message("a", "user", "old")
    db.add_message("b", "user", "question-b")
    db.add_message("a", "user", "question-a")
    db.add_message("a", "tool", "ignored")
    db.add_message("b", "assistant", "answer-b")
    db.add_message("a", "assistant", "answer-a")
    db.add_message("a", "assistant", "orphan")

    assert db.export_all_turns() == [
        {"session_id": "b", "text": "User: question-b\nAssistant: answer-b"},
        {"session_id": "a", "text": "User: question-a\nAssistant: answer-a"},
    ]


def test_desktop_mutation_requires_session_lease(monkeypatch) -> None:
    monkeypatch.setenv("COMPUTER_CONTROL_RATE_LIMIT", "1")
    with activate_session("s"):
        with pytest.raises(PermissionError):
            _authorize_action("computer_click")
        grant_desktop_lease("s")
        _authorize_action("computer_click")
        with pytest.raises(PermissionError, match="rate limit"):
            _authorize_action("computer_click")
        revoke_desktop_lease("s")


def test_screenshot_buffers_are_session_scoped() -> None:
    with activate_session("a"):
        _push_screenshot("image-a")
    with activate_session("b"):
        assert _pop_screenshot() is None
    with activate_session("a"):
        assert _pop_screenshot() == "image-a"


@pytest.mark.asyncio
async def test_unhealthy_mcp_server_attempts_restart(monkeypatch) -> None:
    manager = MCPManager()
    manager._servers["fs"] = MCPServerConfig("fs", "unused", [])
    manager._healthy["fs"] = False
    manager._retry_after["fs"] = 0
    restarted = asyncio.Event()

    async def reconnect(name: str) -> None:
        assert name == "fs"
        manager._healthy[name] = True
        restarted.set()

    monkeypatch.setattr(manager, "_connect_server", reconnect)
    await manager._ping_server("fs")

    assert restarted.is_set()
    assert manager._healthy["fs"] is True
