from __future__ import annotations

from agent.main import _answer_payload, _status_payload


class _MCP:
    tool_names = ["read_file", "write_file"]


class _Graph:
    mcp = _MCP()


def test_status_payload_reports_tools() -> None:
    payload = _status_payload(_Graph())

    assert payload == {
        "status": "ok",
        "tools": 2,
    }


def test_answer_payload_shape() -> None:
    payload = _answer_payload({"text": "done"})

    assert payload == {
        "type": "answer",
        "text": "done",
    }
