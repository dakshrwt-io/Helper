"""Computer control tools using PyAutoGUI for Windows desktop automation.

Destructive actions (click, type, scroll, drag, hotkey) are gated through a
user-confirmation system.  Read-only actions (screenshot, get-position, get-
screen-size) execute immediately.

Confirmation events flow through the trace collector / ChatBus so both the web
UI and Telegram can present allow/deny prompts.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.trace import duration_ms, get_active_trace

logger = logging.getLogger(__name__)

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


# ── Screenshot storage (FIFO) for multimodal injection into LLM ─────
_MAX_SCREENSHOTS = 5
_screenshots: list[str] = []


def _push_screenshot(b64: str) -> None:
    if len(_screenshots) >= _MAX_SCREENSHOTS:
        _screenshots.pop(0)
    _screenshots.append(b64)


def _pop_screenshot() -> str | None:
    if _screenshots:
        return _screenshots.pop(0)
    return None


def inject_screenshots(messages: list[Any]) -> list[Any]:
    """Wrap ``computer_screenshot`` ToolMessages with a follow-up HumanMessage
    carrying the image, as Groq and some vision APIs reject images in ToolMessages.
    """
    result: list[Any] = []
    for m in messages:
        result.append(m)
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "computer_screenshot":
            b64 = _pop_screenshot()
            if b64:
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


# ── Confirmation subsystem ──────────────────────────────────────────
_pending: dict[str, tuple[asyncio.Event, dict[str, Any]]] = {}
_results: dict[str, bool] = {}
_confirm_enabled: bool = True
_confirm_timeout: float = 60.0

READONLY_TOOLS = frozenset(
    {
        "computer_screenshot",
        "computer_get_screen_size",
        "computer_get_mouse_position",
    }
)


def set_confirm_config(*, enabled: bool, timeout: float = 60.0) -> None:
    global _confirm_enabled, _confirm_timeout
    _confirm_enabled = enabled
    _confirm_timeout = timeout


def request_confirmation(tool_name: str, arguments: dict[str, Any]) -> str:
    confirm_id = f"{tool_name}_{int(time.time() * 1000)}"
    event = asyncio.Event()
    _pending[confirm_id] = (event, {"tool_name": tool_name, "arguments": arguments})
    trace = get_active_trace()
    if trace:
        args_summary = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        trace.emit(
            "confirmation_needed",
            confirm_id=confirm_id,
            tool_name=tool_name,
            arguments=arguments,
            description=f"{tool_name}({args_summary})",
        )
    return confirm_id


def resolve_confirmation(confirm_id: str, approved: bool) -> None:
    """Called by web / Telegram handlers to approve or deny a pending action."""
    _results[confirm_id] = approved
    entry = _pending.pop(confirm_id, None)
    if entry:
        event, _ = entry
        event.set()


def get_pending_confirmation(confirm_id: str) -> dict[str, Any] | None:
    entry = _pending.get(confirm_id)
    return entry[1] if entry else None


async def wait_for_confirmation(confirm_id: str) -> bool:
    entry = _pending.get(confirm_id)
    if not entry:
        return False
    event, _ = entry
    try:
        await asyncio.wait_for(event.wait(), timeout=_confirm_timeout)
        return _results.pop(confirm_id, False)
    except asyncio.TimeoutError:
        _pending.pop(confirm_id, None)
        logger.warning("Confirmation %s timed out after %.0fs", confirm_id, _confirm_timeout)
        return False


def _needs_confirm(tool_name: str) -> bool:
    return _confirm_enabled and tool_name not in READONLY_TOOLS


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
    img = pyautogui.screenshot()
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    _push_screenshot(b64)
    return f"Screenshot captured ({img.width}\u00d7{img.height}). Describe what you see."


async def _exec_get_screen_size(**kwargs: Any) -> str:
    _ensure_pyautogui()
    w, h = pyautogui.size()
    return f"Screen size: {w}\u00d7{h} (primary monitor)"


async def _exec_get_mouse_position(**kwargs: Any) -> str:
    _ensure_pyautogui()
    x, y = pyautogui.position()
    return f"Mouse position: ({x}, {y})"


async def _exec_move_mouse(x: int, y: int) -> str:
    if _needs_confirm("computer_move_mouse"):
        cid = request_confirmation("computer_move_mouse", {"x": x, "y": y})
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    pyautogui.moveTo(x, y)
    return f"Moved mouse to ({x}, {y})"


async def _exec_click(x: int, y: int, button: str = "left") -> str:
    if _needs_confirm("computer_click"):
        cid = request_confirmation("computer_click", {"x": x, "y": y, "button": button})
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    pyautogui.click(x, y, button=button)
    return f"Clicked {button} at ({x}, {y})"


async def _exec_double_click(x: int, y: int) -> str:
    if _needs_confirm("computer_double_click"):
        cid = request_confirmation("computer_double_click", {"x": x, "y": y})
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    pyautogui.doubleClick(x, y)
    return f"Double-clicked at ({x}, {y})"


async def _exec_right_click(x: int, y: int) -> str:
    if _needs_confirm("computer_right_click"):
        cid = request_confirmation("computer_right_click", {"x": x, "y": y})
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    pyautogui.rightClick(x, y)
    return f"Right-clicked at ({x}, {y})"


async def _exec_type_text(text: str, interval: float = 0.05) -> str:
    if _needs_confirm("computer_type_text"):
        preview = text if len(text) <= 80 else text[:77] + "..."
        cid = request_confirmation("computer_type_text", {"text": preview})
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    pyautogui.write(text, interval=interval)
    preview = text if len(text) <= 60 else text[:57] + "..."
    return f"Typed: {preview}"


async def _exec_press_key(key: str) -> str:
    if _needs_confirm("computer_press_key"):
        cid = request_confirmation("computer_press_key", {"key": key})
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    pyautogui.press(key)
    return f"Pressed key: {key}"


async def _exec_hotkey(keys: str) -> str:
    if _needs_confirm("computer_hotkey"):
        cid = request_confirmation("computer_hotkey", {"keys": keys})
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    combo = [k.strip().lower() for k in keys.split("+")]
    pyautogui.hotkey(*combo)
    if "win" in combo:
        time.sleep(0.3)
    return f"Pressed hotkey: {keys}"


async def _exec_scroll(clicks: int) -> str:
    if _needs_confirm("computer_scroll"):
        cid = request_confirmation("computer_scroll", {"clicks": clicks})
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    pyautogui.scroll(clicks)
    direction = "up" if clicks > 0 else "down"
    return f"Scrolled {direction} by {abs(clicks)} clicks"


async def _exec_drag(
    start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5
) -> str:
    if _needs_confirm("computer_drag"):
        cid = request_confirmation(
            "computer_drag",
            {"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y},
        )
        if not await wait_for_confirmation(cid):
            return "Action denied by user."
    _ensure_pyautogui()
    pyautogui.moveTo(start_x, start_y)
    pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
    return f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"


# ── Public API ──────────────────────────────────────────────────────


def get_computer_tools() -> list[StructuredTool]:
    """Build and return all computer-control StructuredTools."""
    return [
        # read-only — no confirmation needed
        _wrap_tool(
            "computer_screenshot",
            "Capture a screenshot of the entire primary screen. Always call this "
            "BEFORE any click/type/drag action so you can see what is on screen. "
            "Call it AGAIN after actions to verify the result.",
            BaseModel,
            _exec_screenshot,
        ),
        _wrap_tool(
            "computer_get_screen_size",
            "Get the width and height of the primary monitor in pixels.",
            BaseModel,
            _exec_get_screen_size,
        ),
        _wrap_tool(
            "computer_get_mouse_position",
            "Get the current (x, y) position of the mouse cursor.",
            BaseModel,
            _exec_get_mouse_position,
        ),
        # destructive — may require confirmation
        _wrap_tool(
            "computer_move_mouse",
            "Move the mouse cursor to absolute (x, y) coordinates on the primary "
            "monitor. (0,0) is the top-left corner.",
            _MoveMouseArgs,
            _exec_move_mouse,
        ),
        _wrap_tool(
            "computer_click",
            "Click the left/right/middle mouse button at absolute (x, y) coordinates.",
            _ClickArgs,
            _exec_click,
        ),
        _wrap_tool(
            "computer_double_click",
            "Double-click the left mouse button at absolute (x, y) coordinates.",
            _DoubleClickArgs,
            _exec_double_click,
        ),
        _wrap_tool(
            "computer_right_click",
            "Right-click at absolute (x, y) coordinates (context menu).",
            _RightClickArgs,
            _exec_right_click,
        ),
        _wrap_tool(
            "computer_type_text",
            "Type a string of text as if typed on the keyboard. The target window "
            "must already have focus. Use computer_click first to focus a field.",
            _TypeTextArgs,
            _exec_type_text,
        ),
        _wrap_tool(
            "computer_press_key",
            "Press and release a single keyboard key. Common keys: 'enter', 'esc', "
            "'tab', 'backspace', 'delete', 'home', 'end', 'pageup', 'pagedown', "
            "'up', 'down', 'left', 'right', 'f1'-'f12', 'win', 'alt', 'ctrl'.",
            _PressKeyArgs,
            _exec_press_key,
        ),
        _wrap_tool(
            "computer_hotkey",
            "Press a key combination. Use '+' to separate keys. Examples: "
            "'ctrl+c' (copy), 'ctrl+v' (paste), 'alt+tab' (switch windows), "
            "'win+r' (Run dialog), 'win+d' (show desktop).",
            _HotkeyArgs,
            _exec_hotkey,
        ),
        _wrap_tool(
            "computer_scroll",
            "Scroll the mouse wheel. Positive clicks = scroll up, negative = down. "
            "Typical values: 1-5 clicks for small scrolls, 10+ for longer.",
            _ScrollArgs,
            _exec_scroll,
        ),
        _wrap_tool(
            "computer_drag",
            "Click and drag from (start_x, start_y) to (end_x, end_y). Useful for "
            "moving windows, selecting text, or drag-and-drop operations.",
            _DragArgs,
            _exec_drag,
        ),
    ]
