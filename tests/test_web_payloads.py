from __future__ import annotations

from agent.main import _answer_payload


def test_answer_payload_shape() -> None:
    assert _answer_payload({"text": "done"}) == {
        "type": "answer",
        "text": "done",
    }
