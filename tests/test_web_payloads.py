from __future__ import annotations

from agent.main import _answer_payload, _status_payload


class _MCP:
    tool_names = ["read_file", "write_file"]


class _ChatDB:
    def spent_today(self) -> float:
        return 0.25


class _Graph:
    mcp = _MCP()
    _chatdb = _ChatDB()


def test_status_payload_reports_tools_and_spend(monkeypatch) -> None:
    monkeypatch.setenv("DAILY_COST_USD", "2.5")

    payload = _status_payload(_Graph())

    assert payload == {
        "status": "ok",
        "tools": 2,
        "spent_today": 0.25,
        "daily_cap": 2.5,
    }


def test_answer_payload_keeps_legacy_cost_fields(monkeypatch) -> None:
    monkeypatch.setenv("DAILY_COST_USD", "1.0")

    payload = _answer_payload({"text": "done"}, _Graph())

    assert payload == {
        "type": "answer",
        "text": "done",
        "cost_spent": 0.0,
        "spent_today": 0.25,
        "daily_cap": 1.0,
    }
