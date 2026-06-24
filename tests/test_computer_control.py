"""Unit tests for computer-control tools and confirmation system."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import ToolMessage


# ── helpers ──────────────────────────────────────────────────────────


def _reset_confirmation_state() -> None:
    """Reset module-level state between tests."""
    from agent.tools.computer import (
        _confirm_enabled,
        _pending,
        _results,
        _screenshots,
        set_confirm_config,
    )

    _pending.clear()
    _results.clear()
    _screenshots.clear()
    set_confirm_config(enabled=True, timeout=1.0)


# ── tool creation ────────────────────────────────────────────────────


def test_all_tools_created() -> None:
    _reset_confirmation_state()
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
    _reset_confirmation_state()
    from langchain_core.tools import StructuredTool

    from agent.tools.computer import get_computer_tools

    for tool in get_computer_tools():
        assert isinstance(tool, StructuredTool)
        assert tool.name.startswith("computer_")
        assert tool.description


# ── screenshot injection ─────────────────────────────────────────────


def test_inject_screenshots_replaces_tool_message() -> None:
    _reset_confirmation_state()
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
    _reset_confirmation_state()
    from agent.tools.computer import inject_screenshots

    original = ToolMessage(
        content="Screenshot captured.",
        tool_call_id="call_1",
        name="computer_screenshot",
    )
    result = inject_screenshots([original])
    assert result[0] is original


def test_inject_screenshots_skips_non_screenshot_messages() -> None:
    _reset_confirmation_state()
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


# ── confirmation flow ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_confirmation_approve() -> None:
    _reset_confirmation_state()
    from agent.tools.computer import (
        request_confirmation,
        resolve_confirmation,
        wait_for_confirmation,
    )

    cid = request_confirmation("computer_click", {"x": 10, "y": 20})

    async def _approve() -> None:
        await asyncio.sleep(0.05)
        resolve_confirmation(cid, True)

    task = asyncio.create_task(_approve())
    result = await wait_for_confirmation(cid)
    await task

    assert result is True


@pytest.mark.asyncio
async def test_confirmation_deny() -> None:
    _reset_confirmation_state()
    from agent.tools.computer import (
        request_confirmation,
        resolve_confirmation,
        wait_for_confirmation,
    )

    cid = request_confirmation("computer_type_text", {"text": "hello"})
    resolve_confirmation(cid, False)
    result = await wait_for_confirmation(cid)
    assert result is False


@pytest.mark.asyncio
async def test_confirmation_timeout() -> None:
    _reset_confirmation_state()
    from agent.tools.computer import wait_for_confirmation, request_confirmation

    cid = request_confirmation("computer_click", {"x": 1, "y": 1})
    result = await wait_for_confirmation(cid)
    assert result is False


@pytest.mark.asyncio
async def test_unknown_confirm_id_returns_false() -> None:
    _reset_confirmation_state()
    from agent.tools.computer import wait_for_confirmation

    result = await wait_for_confirmation("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_get_pending_confirmation() -> None:
    _reset_confirmation_state()
    from agent.tools.computer import get_pending_confirmation, request_confirmation

    cid = request_confirmation("computer_click", {"x": 42, "y": 99})
    info = get_pending_confirmation(cid)
    assert info is not None
    assert info["tool_name"] == "computer_click"
    assert info["arguments"] == {"x": 42, "y": 99}

    info_missing = get_pending_confirmation("bogus")
    assert info_missing is None


# ── confirmation gating ──────────────────────────────────────────────


def test_readonly_tools_skip_confirmation() -> None:
    _reset_confirmation_state()
    from agent.tools.computer import READONLY_TOOLS, _needs_confirm

    for name in READONLY_TOOLS:
        assert _needs_confirm(name) is False


def test_destructive_tools_require_confirmation_when_enabled() -> None:
    _reset_confirmation_state()
    from agent.tools.computer import _needs_confirm

    destructive = [
        "computer_click",
        "computer_type_text",
        "computer_hotkey",
        "computer_scroll",
        "computer_drag",
        "computer_move_mouse",
    ]
    for name in destructive:
        assert _needs_confirm(name) is True


def test_confirm_disabled_skips_all() -> None:
    _reset_confirmation_state()
    from agent.tools.computer import _needs_confirm, set_confirm_config

    set_confirm_config(enabled=False)
    for name in ["computer_click", "computer_type_text", "computer_hotkey"]:
        assert _needs_confirm(name) is False


def test_set_confirm_config_timeout() -> None:
    _reset_confirmation_state()
    from agent.tools import computer as cc

    cc.set_confirm_config(enabled=True, timeout=30.0)
    assert cc._confirm_timeout == 30.0
