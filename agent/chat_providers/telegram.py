"""Telegram bot entry point: receive messages, forward to AgentGraph, reply."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.helpers import escape_markdown

from agent.shared import get_graph, get_chat_lock, get_chat_bus, get_cancel_event, request_cancel
from agent.tools.computer import grant_desktop_lease, revoke_desktop_lease

logger = logging.getLogger("agent.telegram")
_MAX_TELEGRAM_LEN = 4000


def _parse_allowed_users(raw: str | None) -> set[int]:
    if not raw or not raw.strip():
        return set()
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            ids.add(int(part))
    return ids


def _check_access(update: Update) -> tuple[bool, int | None]:
    """Return (authorized, user_id). user_id is None if not available."""
    allowed = _parse_allowed_users(os.environ.get("TELEGRAM_ALLOWED_USERS", ""))
    user_id = update.effective_user.id if update.effective_user else None
    if not allowed:
        logger.warning("TELEGRAM_ALLOWED_USERS is empty — access denied")
        return False, user_id
    if user_id is None:
        return False, None
    if user_id not in allowed:
        logger.info("Access denied for Telegram user %d", user_id)
        return False, user_id
    return True, user_id


def _split_by_length(text: str, limit: int, chunks: list[str]) -> None:
    """Split a long run-on string by byte length into chunks."""
    b = text.encode("utf-8")
    pos = 0
    while pos < len(b):
        end = min(pos + limit, len(b))
        chunk = b[pos:end].decode("utf-8", errors="ignore")
        chunks.append(chunk)
        pos = end


def _split_overlong(text: str, limit: int, chunks: list[str]) -> str:
    """Handle a single sentence that exceeds the limit: split by length, return leftover."""
    b = text.encode("utf-8")
    if len(b) <= limit:
        return text
    pos = 0
    while pos + limit < len(b):
        chunks.append(b[pos:pos + limit].decode("utf-8", errors="ignore"))
        pos += limit
    return b[pos:].decode("utf-8", errors="ignore")


def _split_message(text: str, limit: int = _MAX_TELEGRAM_LEN) -> list[str]:
    """Split at paragraph then sentence boundaries so Telegram limit is respected."""
    if len(text.encode("utf-8")) <= limit:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para.encode("utf-8")) <= limit:
            chunks.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            if len(sentences) <= 1:
                _split_by_length(para, limit, chunks)
            else:
                buf = ""
                for s in sentences:
                    candidate = (buf + " " + s).strip() if buf else s
                    if len(candidate.encode("utf-8")) <= limit:
                        buf = candidate
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = _split_overlong(s, limit, chunks)
                if buf:
                    chunks.append(buf)

    if not chunks:
        return [text[:limit]]

    total = len(chunks)
    if total == 1:
        return chunks
    return [f"[{i + 1}/{total}]\n{c}" for i, c in enumerate(chunks)]


async def _send_long(update: Update, text: str) -> None:
    """Send arbitrary assistant text, split into Telegram-sized messages.

    Model output is not guaranteed to be valid Telegram MarkdownV2. Sending it
    as plain text avoids parse errors that would otherwise drop the final reply.
    """
    parts = _split_message(text)
    for i, part in enumerate(parts):
        await _send_with_retry(update, part, parse_mode=None)
        if i < len(parts) - 1:
            await asyncio.sleep(0.05)


async def _send_safe(update: Update, text: str) -> None:
    """Send a single message with MarkdownV2, falling back to plain text."""
    try:
        await _send_with_retry(update, text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception:
        await _send_with_retry(update, text, parse_mode=None)


async def _send_with_retry(
    update: Update,
    text: str,
    parse_mode: str | None = None,
    max_retries: int = 3,
) -> None:
    for attempt in range(max_retries):
        try:
            await update.effective_chat.send_message(text, parse_mode=parse_mode)
            return
        except RetryAfter as e:
            wait = float(e.retry_after) + 0.5
            logger.warning(
                "Telegram 429 rate limit hit, retrying after %.1fs (attempt %d/%d)",
                wait, attempt + 1, max_retries,
            )
            await asyncio.sleep(wait)
        except (TimedOut, NetworkError):
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("Max retries exceeded for Telegram send")


# ── command handlers ──────────────────────────────────────────────


async def start_cmd(update: Update, _context: Any) -> None:
    if not update.effective_chat:
        return
    authorized, tg_id = _check_access(update)
    if not authorized:
        await _send_safe(update, f"\U0001f512 *Access denied*\nYour Telegram user ID: `{tg_id or 'unknown'}`\nAdd it to TELEGRAM_ALLOWED_USERS in your \\.env file\\.")
        return
    graph = get_graph()
    tool_count = len(graph.mcp.tool_names) if graph and graph.mcp else 0
    await _send_safe(
        update,
        f"Hi {update.effective_user.first_name}\\!\n\n"
        f"I am your personal AI agent\\. I can browse the web, read and write files, "
        f"search documentation, and run code\\.\n\n"
        f"*Available tools:* {tool_count}\n"
        f"*Model:* {graph._model_name if graph else '?'}\n\n"
        f"Just send me any request to get started\\.\n"
        f"*Commands:* /help \\| /reset",
    )


async def help_cmd(update: Update, _context: Any) -> None:
    if not update.effective_chat:
        return
    authorized, tg_id = _check_access(update)
    if not authorized:
        await _send_safe(update, f"\U0001f512 *Access denied*\nYour Telegram user ID: `{tg_id or 'unknown'}`\nAdd it to TELEGRAM_ALLOWED_USERS in your \\.env file\\.")
        return
    await _send_safe(
        update,
        "*Commands*\n\n"
        "/start  \\- Welcome message\n"
        "/help   \\- This list\n"
        "/reset  \\- Start a fresh conversation\n\n"
        "You can also just send me any message and I will help you\\.",
    )


async def reset_cmd(update: Update, _context: Any) -> None:
    if not update.effective_chat:
        return
    authorized, tg_id = _check_access(update)
    if not authorized:
        await _send_safe(update, f"\U0001f512 *Access denied*\nYour Telegram user ID: `{tg_id or 'unknown'}`\nAdd it to TELEGRAM_ALLOWED_USERS in your \\.env file\\.")
        return
    old_session = _context.user_data.pop("session_id", None)
    if old_session:
        revoke_desktop_lease(old_session)
    await _send_safe(update, "Conversation reset\\. The agent will not remember previous messages from this chat\\.")


async def desktop_on_cmd(update: Update, _context: Any) -> None:
    authorized, _ = _check_access(update)
    if not authorized or not update.effective_chat:
        return
    session_id = _context.user_data.get("session_id") or f"telegram_{update.effective_chat.id}"
    _context.user_data["session_id"] = session_id
    seconds = grant_desktop_lease(session_id)
    await _send_safe(update, f"Desktop control enabled for {seconds} seconds\\.")


async def desktop_off_cmd(update: Update, _context: Any) -> None:
    authorized, _ = _check_access(update)
    if not authorized or not update.effective_chat:
        return
    session_id = _context.user_data.get("session_id") or f"telegram_{update.effective_chat.id}"
    revoke_desktop_lease(session_id)
    await _send_safe(update, "Desktop control disabled\\.")


async def stop_cmd(update: Update, _context: Any) -> None:
    authorized, _ = _check_access(update)
    if not authorized or not update.effective_chat:
        return
    session_id = _context.user_data.get("session_id") or f"telegram_{update.effective_chat.id}"
    request_cancel(session_id)
    await _send_safe(update, "Cancelling current task\\.\\.\\.")


# ── message handler ───────────────────────────────────────────────


async def message_handler(update: Update, _context: Any) -> None:
    if not update.effective_chat or not update.message or not update.message.text:
        return

    authorized, tg_id = _check_access(update)
    if not authorized:
        await _send_safe(update, f"\U0001f512 *Access denied*\nYour Telegram user ID: `{tg_id or 'unknown'}`\nAdd it to TELEGRAM_ALLOWED_USERS in your \\.env file\\.")
        return

    graph = get_graph()
    if graph is None:
        await _send_safe(update, "Agent is still starting up\\. Please try again in a moment\\.")
        return

    user_text = update.message.text.strip()
    chat_id = update.effective_chat.id
    try:
        session_id = _context.user_data.get("session_id")
    except Exception:
        session_id = None
    if not session_id:
        session_id = f"telegram_{chat_id}"
        _context.user_data["session_id"] = session_id

    bus = get_chat_bus()
    trace_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def _bridge() -> None:
        try:
            while True:
                event = await trace_queue.get()
                await bus.publish(session_id, {"type": "trace", "event": event})
        except asyncio.CancelledError:
            raise

    bridge_task = asyncio.create_task(_bridge())

    try:
        await bus.publish(session_id, {"type": "remote_user", "text": user_text, "source": "telegram"})
        await bus.publish(session_id, {"type": "thinking"})

        async def _locked_chat() -> dict:
            async with get_chat_lock(session_id):
                return await graph.chat(user_text, session_id=session_id, trace_queue=trace_queue, cancel_event=get_cancel_event(session_id))

        try:
            await update.effective_chat.send_chat_action("typing")
        except Exception:
            pass
        result = await asyncio.wait_for(_locked_chat(), timeout=180)
    except asyncio.TimeoutError:
        bridge_task.cancel()
        try:
            await bridge_task
        except asyncio.CancelledError:
            pass
        while not trace_queue.empty():
            try:
                await bus.publish(session_id, {"type": "trace", "event": trace_queue.get_nowait()})
            except asyncio.QueueEmpty:
                break
        await _send_safe(update, "The agent timed out \\(180s\\)\\. Please try a simpler request\\.")
        return
    except Exception as exc:
        bridge_task.cancel()
        try:
            await bridge_task
        except asyncio.CancelledError:
            pass
        logger.exception("Telegram handler error")
        await _send_safe(update, f"*Error*: {exc}")
        return

    bridge_task.cancel()
    try:
        await bridge_task
    except asyncio.CancelledError:
        pass

    while not trace_queue.empty():
        try:
            await bus.publish(session_id, {"type": "trace", "event": trace_queue.get_nowait()})
        except asyncio.QueueEmpty:
            break

    final_text = result.get("text", "").strip()
    if not final_text:
        await _send_safe(update, "I did not produce a final answer\\. Please try rephrasing your request\\.")
        return

    await bus.publish(session_id, {
        "type": "answer",
        "text": final_text,
    })

    await _send_long(update, final_text)


# ── builder ───────────────────────────────────────────────────────


def build_bot(token: str) -> Application:
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("desktop_on", desktop_on_cmd))
    app.add_handler(CommandHandler("desktop_off", desktop_off_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Telegram bot handlers registered")
    return app
