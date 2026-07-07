from __future__ import annotations

from agent.memory.db import ChatDB


def test_add_turn_persists_user_then_assistant() -> None:
    db = ChatDB(":memory:")

    db.add_turn("session-1", "hello", "hi there")

    history = db.get_history("session-1")

    assert [row["role"] for row in history] == ["user", "assistant"]
    assert [row["content"] for row in history] == ["hello", "hi there"]
    assert all(row["ts"] for row in history)
