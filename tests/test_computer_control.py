"""Unit tests for computer-control tools."""
from __future__ import annotations

from langchain_core.messages import ToolMessage


# ── helpers ──────────────────────────────────────────────────────────


def _reset_state() -> None:
    """Reset module-level state between tests."""
    from agent.tools.computer import _screenshots

    _screenshots.clear()


# ── tool creation ────────────────────────────────────────────────────


def test_all_tools_created() -> None:
    _reset_state()
    from agent.tools.computer import get_computer_tools

    tools = get_computer_tools()
    names = {t.name for t in tools}

    assert len(tools) == 12
    assert "computer_screenshot" in names
    assert "computer_get_screen_size" in names
    assert "computer_get_mouse_position" in names
    assert "computer_move_mouse" in names
    assert "computer_click" in names
    assert "computer_double_click" in names
    assert "computer_right_click" in names
    assert "computer_type_text" in names
    assert "computer_press_key" in names
    assert "computer_hotkey" in names
    assert "computer_scroll" in names
    assert "computer_drag" in names


def test_all_tools_are_structured() -> None:
    _reset_state()
    from langchain_core.tools import StructuredTool

    from agent.tools.computer import get_computer_tools

    for tool in get_computer_tools():
        assert isinstance(tool, StructuredTool)
        assert tool.name.startswith("computer_")
        assert tool.description


# ── screenshot injection ─────────────────────────────────────────────


def test_inject_screenshots_replaces_tool_message() -> None:
    _reset_state()
    from agent.tools.computer import _push_screenshot, inject_screenshots

    b64 = "iVBORw0KGgo="
    _push_screenshot(b64)

    original = ToolMessage(
        content="Screenshot captured (800\u00d7600).",
        tool_call_id="call_1",
        name="computer_screenshot",
    )
    result = inject_screenshots([original])

    assert len(result) == 2
    assert result[0] is original
    image_msg = result[1]
    content = image_msg.content
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == f"data:image/png;base64,{b64}"
    assert content[1]["type"] == "text"


def test_inject_screenshots_no_image_keeps_original() -> None:
    _reset_state()
    from agent.tools.computer import inject_screenshots

    original = ToolMessage(
        content="Screenshot captured.",
        tool_call_id="call_1",
        name="computer_screenshot",
    )
    result = inject_screenshots([original])
    assert result[0] is original


def test_inject_screenshots_skips_non_screenshot_messages() -> None:
    _reset_state()
    from agent.tools.computer import _push_screenshot, inject_screenshots

    _push_screenshot("b64")
    msgs = [
        ToolMessage(content="done", tool_call_id="c1", name="computer_click"),
        ToolMessage(content="ok", tool_call_id="c2", name="computer_screenshot"),
    ]
    result = inject_screenshots(msgs)
    assert len(result) == 3
    assert result[0].content == "done"
    assert result[1].content == "ok"
    assert isinstance(result[2].content, list)
