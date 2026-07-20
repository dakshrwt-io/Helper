"""Computer control tools using PyAutoGUI for Windows desktop automation.

All actions (screenshot, mouse, keyboard, scroll, drag) execute immediately.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from collections import defaultdict, deque
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from PIL import Image
from pydantic import BaseModel, Field

from agent.shared import get_active_session_id, sanitize_aimessage
from agent.trace import duration_ms, get_active_trace

# ── PyAutoGUI availability ──────────────────────────────────────────
_pyautogui_ok = True
try:
    import pyautogui

    pyautogui.PAUSE = 0.1
    pyautogui.FAILSAFE = True
except Exception:
    _pyautogui_ok = False


def _ensure_pyautogui() -> None:
    if not _pyautogui_ok:
        raise RuntimeError(
            "PyAutoGUI is not available. Install with: pip install PyAutoGUI Pillow"
        )


def _annotate_screenshot(img: Image.Image) -> Image.Image:
    """Overlay coordinate grid with axis labels for visual grounding.

    Gridlines every 100px with red axis labels at (x, 0) and (0, y).
    Best-effort — silently returns original image on any error.
    """
    try:
        from PIL import ImageDraw, ImageFont

        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()

        grid_color = (255, 60, 60, 50)  # semi-transparent red
        for x in range(0, w, 100):
            draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
        for y in range(0, h, 100):
            draw.line([(0, y), (w, y)], fill=grid_color, width=1)

        label_color = (255, 60, 60, 180)
        for x in range(0, w, 100):
            draw.text((x + 2, 2), str(x), fill=label_color, font=font)
        for y in range(100, h, 100):
            draw.text((2, y + 2), str(y), fill=label_color, font=font)

        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")
    except Exception:
        return img


# ── Screenshot storage (FIFO) for multimodal injection into LLM ─────
_MAX_SCREENSHOTS = 5
_screenshots: dict[str, list[str]] = defaultdict(list)


def _screenshot_buffer() -> list[str]:
    return _screenshots[get_active_session_id() or "__unscoped__"]


def _push_screenshot(b64: str) -> None:
    screenshots = _screenshot_buffer()
    if len(screenshots) >= _MAX_SCREENSHOTS:
        screenshots.pop(0)
    screenshots.append(b64)


def _pop_screenshot() -> str | None:
    screenshots = _screenshot_buffer()
    if screenshots:
        return screenshots.pop(0)
    return None


def clear_screenshots() -> None:
    _screenshots.pop(get_active_session_id() or "__unscoped__", None)
    return None


READ_ONLY_TOOLS = frozenset(
    {
        "computer_screenshot",
        "computer_get_screen_size",
        "computer_get_mouse_position",
    }
)
_desktop_leases: dict[str, float] = {}
_desktop_actions: dict[str, deque[float]] = defaultdict(deque)


def grant_desktop_lease(session_id: str) -> int:
    seconds = min(
        300, max(1, int(os.environ.get("COMPUTER_CONTROL_LEASE_SECONDS", "300")))
    )
    _desktop_leases[session_id] = time.monotonic() + seconds
    _desktop_actions.pop(session_id, None)
    return seconds


def revoke_desktop_lease(session_id: str) -> None:
    _desktop_leases.pop(session_id, None)
    _desktop_actions.pop(session_id, None)


def _authorize_action(tool_name: str) -> None:
    if tool_name in READ_ONLY_TOOLS:
        return
    session_id = get_active_session_id()
    if not session_id or _desktop_leases.get(session_id, 0) <= time.monotonic():
        if session_id:
            revoke_desktop_lease(session_id)
        raise PermissionError("Desktop control requires an active user-approved lease")

    limit = max(1, int(os.environ.get("COMPUTER_CONTROL_RATE_LIMIT", "30")))
    now = time.monotonic()
    actions = _desktop_actions[session_id]
    while actions and actions[0] <= now - 60:
        actions.popleft()
    if len(actions) >= limit:
        raise PermissionError("Desktop control rate limit exceeded")
    actions.append(now)


def strip_image_content(messages: list[Any]) -> list[Any]:
    """Remove all image_url content blocks from messages.

    After vision LLM processes screenshots, the acting LLM may be text-only
    and reject image_url blocks. This strips them while preserving text.
    """
    result: list[Any] = []
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, list):
            # Keep only text blocks; drop image_url blocks
            text_parts = [
                block for block in content
                if isinstance(block, dict) and block.get("type") != "image_url"
            ]
            if text_parts:
                # Rebuild message with text-only content
                new_content = text_parts if len(text_parts) > 1 else text_parts[0].get("text", "")
                try:
                    result.append(m.__class__(content=new_content))
                except Exception:
                    result.append(m)
            else:
                # All content was images — drop the message entirely
                pass
        else:
            result.append(m)
    return result


def inject_screenshots(messages: list[Any]) -> list[Any]:
    """Insert HumanMessage(image_url) after each computer_screenshot ToolMessage.

    Non-destructive — reads screenshot buffer without consuming it.
    Use clear_screenshots() to reset the buffer between turns.
    """
    result: list[Any] = []
    screenshots = list(_screenshot_buffer())  # copy, don't consume
    ss_idx = 0
    for m in messages:
        clean = sanitize_aimessage(m) if isinstance(m, AIMessage) else m
        result.append(clean)
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "computer_screenshot":
            if ss_idx < len(screenshots):
                b64 = screenshots[ss_idx]
                ss_idx += 1
                result.append(
                    HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {"type": "text", "text": str(m.content)},
                        ],
                    )
                )
    return result


# ── Trace-aware tool wrapper ────────────────────────────────────────
def _wrap_tool(
    name: str,
    desc: str,
    args_schema: type[BaseModel],
    execute: Any,
) -> StructuredTool:
    async def _run(**kwargs: Any) -> str:
        trace = get_active_trace()
        start = time.perf_counter()
        clean_args = {k: v for k, v in kwargs.items() if v is not None}
        if trace:
            trace.emit("tool_started", tool_name=name, server="computer", arguments=clean_args)
        try:
            _authorize_action(name)
            result = await execute(**clean_args)
        except Exception as exc:
            if trace:
                trace.emit(
                    "tool_failed",
                    tool_name=name,
                    server="computer",
                    duration_ms=duration_ms(start),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise
        if trace:
            trace.emit(
                "tool_completed",
                tool_name=name,
                server="computer",
                duration_ms=duration_ms(start),
                output=result,
            )
        return result

    return StructuredTool.from_function(
        coroutine=_run,
        name=name,
        description=desc,
        args_schema=args_schema,
    )


# ── Pydantic arg schemas ────────────────────────────────────────────


class _ClickArgs(BaseModel):
    x: int = Field(..., description="X coordinate on screen")
    y: int = Field(..., description="Y coordinate on screen")
    button: str = Field(
        default="left",
        description="Mouse button: 'left', 'right', or 'middle'",
    )


class _MoveMouseArgs(BaseModel):
    x: int = Field(..., description="X coordinate to move to")
    y: int = Field(..., description="Y coordinate to move to")


class _TypeTextArgs(BaseModel):
    text: str = Field(..., description="Text to type")
    interval: float = Field(default=0.05, description="Seconds between keystrokes")


class _PressKeyArgs(BaseModel):
    key: str = Field(..., description="Key name, e.g. 'enter', 'esc', 'tab', 'a'")


class _HotkeyArgs(BaseModel):
    keys: str = Field(
        ...,
        description="Key combination separated by '+', e.g. 'ctrl+c', 'alt+tab'",
    )


class _ScrollArgs(BaseModel):
    clicks: int = Field(
        ..., description="Positive = scroll up, negative = scroll down"
    )


class _DragArgs(BaseModel):
    start_x: int = Field(..., description="Starting X coordinate")
    start_y: int = Field(..., description="Starting Y coordinate")
    end_x: int = Field(..., description="Ending X coordinate")
    end_y: int = Field(..., description="Ending Y coordinate")
    duration: float = Field(default=0.5, description="Drag duration in seconds")


class _DoubleClickArgs(BaseModel):
    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")


class _RightClickArgs(BaseModel):
    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")


# ── Tool implementations ────────────────────────────────────────────


async def _exec_screenshot(**kwargs: Any) -> str:
    _ensure_pyautogui()
    img = await asyncio.to_thread(pyautogui.screenshot)
    img = _annotate_screenshot(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    _push_screenshot(b64)
    return (
        f"Screenshot captured ({img.width}\u00d7{img.height}) with coordinate grid. "
        f"Gridlines every 100px. Red labels at edges show (x,y) coordinates. "
        f"Use grid labels to estimate click targets precisely."
    )


async def _exec_get_screen_size(**kwargs: Any) -> str:
    _ensure_pyautogui()
    w, h = await asyncio.to_thread(pyautogui.size)
    return f"Screen size: {w}\u00d7{h} (primary monitor)"


async def _exec_get_mouse_position(**kwargs: Any) -> str:
    _ensure_pyautogui()
    x, y = await asyncio.to_thread(pyautogui.position)
    return f"Mouse position: ({x}, {y})"


async def _exec_move_mouse(x: int, y: int) -> str:
    _ensure_pyautogui()
    await asyncio.to_thread(pyautogui.moveTo, x, y)
    return f"Moved mouse to ({x}, {y})"


async def _exec_click(x: int, y: int, button: str = "left") -> str:
    _ensure_pyautogui()
    await asyncio.to_thread(pyautogui.click, x, y, button=button)
    return f"Clicked {button} at ({x}, {y})"


async def _exec_double_click(x: int, y: int) -> str:
    _ensure_pyautogui()
    await asyncio.to_thread(pyautogui.doubleClick, x, y)
    return f"Double-clicked at ({x}, {y})"


async def _exec_right_click(x: int, y: int) -> str:
    _ensure_pyautogui()
    await asyncio.to_thread(pyautogui.rightClick, x, y)
    return f"Right-clicked at ({x}, {y})"


async def _exec_type_text(text: str, interval: float = 0.05) -> str:
    _ensure_pyautogui()
    await asyncio.to_thread(pyautogui.write, text, interval=interval)
    preview = text if len(text) <= 60 else text[:57] + "..."
    return f"Typed: {preview}"


async def _exec_press_key(key: str) -> str:
    _ensure_pyautogui()
    await asyncio.to_thread(pyautogui.press, key)
    return f"Pressed key: {key}"


async def _exec_hotkey(keys: str) -> str:
    _ensure_pyautogui()
    combo = [k.strip().lower() for k in keys.split("+")]
    await asyncio.to_thread(pyautogui.hotkey, *combo)
    if "win" in combo:
        await asyncio.sleep(0.3)
    return f"Pressed hotkey: {keys}"


async def _exec_scroll(clicks: int) -> str:
    _ensure_pyautogui()
    await asyncio.to_thread(pyautogui.scroll, clicks)
    direction = "up" if clicks > 0 else "down"
    return f"Scrolled {direction} by {abs(clicks)} clicks"


async def _exec_drag(
    start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5
) -> str:
    _ensure_pyautogui()
    await asyncio.to_thread(pyautogui.moveTo, start_x, start_y)
    await asyncio.to_thread(pyautogui.drag, end_x - start_x, end_y - start_y, duration=duration)
    return f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"


# ── Public API ──────────────────────────────────────────────────────


def get_computer_tools() -> list[StructuredTool]:
    """Build and return all computer-control StructuredTools."""
    specs = [
        (
            "computer_screenshot",
            "Capture a screenshot of the entire primary screen. Use this when you need "
            "to see what is on screen — identifying UI elements, verifying an action "
            "completed, or checking the current state.",
            BaseModel,
            _exec_screenshot,
        ),
        (
            "computer_get_screen_size",
            "Get the width and height of the primary monitor in pixels.",
            BaseModel,
            _exec_get_screen_size,
        ),
        (
            "computer_get_mouse_position",
            "Get the current (x, y) position of the mouse cursor.",
            BaseModel,
            _exec_get_mouse_position,
        ),
        (
            "computer_move_mouse",
            "Move the mouse cursor to absolute (x, y) coordinates on the primary "
            "monitor. (0,0) is the top-left corner.",
            _MoveMouseArgs,
            _exec_move_mouse,
        ),
        (
            "computer_click",
            "Click the left/right/middle mouse button at absolute (x, y) coordinates.",
            _ClickArgs,
            _exec_click,
        ),
        (
            "computer_double_click",
            "Double-click the left mouse button at absolute (x, y) coordinates.",
            _DoubleClickArgs,
            _exec_double_click,
        ),
        (
            "computer_right_click",
            "Right-click at absolute (x, y) coordinates (context menu).",
            _RightClickArgs,
            _exec_right_click,
        ),
        (
            "computer_type_text",
            "Type a string of text as if typed on the keyboard. The target window "
            "must already have focus. Use computer_click first to focus a field.",
            _TypeTextArgs,
            _exec_type_text,
        ),
        (
            "computer_press_key",
            "Press and release a single keyboard key. Common keys: 'enter', 'esc', "
            "'tab', 'backspace', 'delete', 'home', 'end', 'pageup', 'pagedown', "
            "'up', 'down', 'left', 'right', 'f1'-'f12', 'win', 'alt', 'ctrl'.",
            _PressKeyArgs,
            _exec_press_key,
        ),
        (
            "computer_hotkey",
            "Press a key combination. Use '+' to separate keys. Examples: "
            "'ctrl+c' (copy), 'ctrl+v' (paste), 'alt+tab' (switch windows), "
            "'win+r' (Run dialog), 'win+d' (show desktop).",
            _HotkeyArgs,
            _exec_hotkey,
        ),
        (
            "computer_scroll",
            "Scroll the mouse wheel. Positive clicks = scroll up, negative = down. "
            "Typical values: 1-5 clicks for small scrolls, 10+ for longer.",
            _ScrollArgs,
            _exec_scroll,
        ),
        (
            "computer_drag",
            "Click and drag from (start_x, start_y) to (end_x, end_y). Useful for "
            "moving windows, selecting text, or drag-and-drop operations.",
            _DragArgs,
            _exec_drag,
        ),
    ]
    return [_wrap_tool(*spec) for spec in specs]
