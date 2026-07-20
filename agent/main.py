"""FastAPI server: WebSocket /chat + static HTML frontend."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from agent.shared import get_graph, get_chat_lock, get_chat_bus, get_cancel_event, request_cancel
from agent.tools.computer import grant_desktop_lease, revoke_desktop_lease

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent.web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _web_token():
        raise RuntimeError("WEB_TOKEN must be set")
    graph = get_graph()
    if graph:
        logger.info("AgentGraph ready. tools=%d", len(graph.mcp.tool_names) if graph.mcp else 0)
    yield
    logger.info("Web server shutting down")


app = FastAPI(lifespan=lifespan)
_COOKIE_NAME = "helper_session"
_login_attempts: dict[str, deque[float]] = defaultdict(deque)


def _web_token() -> str:
    return os.environ.get("WEB_TOKEN", "")


def _sign_session(session_id: str) -> str:
    signature = hmac.new(
        _web_token().encode(), session_id.encode(), hashlib.sha256
    ).hexdigest()
    return f"{session_id}.{signature}"


def _verify_session(value: str | None) -> str | None:
    if not value or "." not in value or not _web_token():
        return None
    session_id, signature = value.rsplit(".", 1)
    expected = _sign_session(session_id).rsplit(".", 1)[1]
    if not hmac.compare_digest(signature, expected):
        return None
    return session_id if session_id.startswith("web_") else None


def _origin_allowed(ws: WebSocket) -> bool:
    origin = ws.headers.get("origin")
    if not origin:
        return bool(ws.headers.get("authorization"))
    allowed = {
        value.strip()
        for value in os.environ.get("WEB_ALLOWED_ORIGINS", "").split(",")
        if value.strip()
    }
    if origin in allowed:
        return True
    parsed = urlparse(origin)
    return parsed.netloc == ws.headers.get("host")


def _authenticate_ws(ws: WebSocket) -> str | None:
    session_id = _verify_session(ws.cookies.get(_COOKIE_NAME))
    if session_id:
        return session_id
    auth = ws.headers.get("authorization", "")
    if auth.startswith("Bearer ") and hmac.compare_digest(
        auth[7:], _web_token()
    ):
        return f"web_{secrets.token_hex(16)}"
    return None


def _tool_count(graph) -> int:
    return len(graph.mcp.tool_names) if graph and graph.mcp else 0


def _status_payload(graph=None) -> dict:
    graph = get_graph() if graph is None else graph
    return {
        "status": "ok",
        "tools": _tool_count(graph),
    }


def _answer_payload(result: dict) -> dict:
    return {
        "type": "answer",
        "text": result["text"],
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "web", "index.html"))


@app.post("/auth/login")
async def login(request: Request):
    now = time.monotonic()
    address = request.client.host if request.client else "unknown"
    attempts = _login_attempts[address]
    while attempts and attempts[0] < now - 60:
        attempts.popleft()
    if len(attempts) >= 5:
        return JSONResponse({"error": "Too many attempts"}, status_code=429)
    attempts.append(now)

    try:
        supplied = str((await request.json()).get("token", ""))
    except (json.JSONDecodeError, AttributeError):
        supplied = ""
    expected = _web_token()
    if not expected or not hmac.compare_digest(supplied, expected):
        return JSONResponse({"error": "Invalid token"}, status_code=401)

    attempts.clear()
    session_id = f"web_{secrets.token_hex(16)}"
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        _COOKIE_NAME,
        _sign_session(session_id),
        httponly=True,
        samesite="strict",
        secure=os.environ.get("WEB_COOKIE_SECURE", "").lower() == "true",
        max_age=86400,
    )
    return response


@app.post("/auth/logout")
async def logout(request: Request):
    session_id = _verify_session(request.cookies.get(_COOKIE_NAME))
    if session_id:
        revoke_desktop_lease(session_id)
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(_COOKIE_NAME)
    return response


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


async def _drain_trace_queue(
    ws: WebSocket,
    trace_queue: asyncio.Queue[dict],
    *,
    suppress_send_errors: bool = False,
) -> None:
    """Forward all currently queued trace events to the WebSocket."""
    while not trace_queue.empty():
        if suppress_send_errors:
            try:
                await ws.send_json({"type": "trace", "event": trace_queue.get_nowait()})
            except Exception:
                break
        else:
            await ws.send_json({"type": "trace", "event": trace_queue.get_nowait()})


async def _wait_for_first(
    primary_task: asyncio.Task,
    *watch_tasks: asyncio.Task | None,
    timeout: float | None = None,
) -> set[asyncio.Task]:
    """Wait for the primary task or one of its existing watcher tasks."""
    waiting = {primary_task}
    waiting.update(task for task in watch_tasks if task is not None)
    done, _ = await asyncio.wait(
        waiting,
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )
    return done


async def _run_chat_with_trace(ws: WebSocket, user_text: str, session_id: str, cancel_event: asyncio.Event | None = None) -> dict:
    """Run a turn while forwarding its internal execution events to the client."""
    graph = get_graph()
    trace_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def _locked_chat():
        async with get_chat_lock(session_id):
            return await graph.chat(user_text, session_id=session_id, trace_queue=trace_queue, cancel_event=cancel_event)

    chat_task = asyncio.create_task(_locked_chat())
    event_task: asyncio.Task | None = asyncio.create_task(trace_queue.get())
    deadline = asyncio.get_running_loop().time() + 180

    # Watch cancel event and cancel chat_task when fired
    cancel_watch: asyncio.Task | None = None
    if cancel_event:
        async def _watch_cancel():
            await cancel_event.wait()
        cancel_watch = asyncio.create_task(_watch_cancel())

    try:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            done = await _wait_for_first(
                chat_task,
                event_task,
                cancel_watch,
                timeout=remaining,
            )
            if not done:
                raise asyncio.TimeoutError

            if cancel_watch is not None and cancel_watch in done:
                chat_task.cancel()
                # Drain remaining trace events so client gets clean state
                await _drain_trace_queue(ws, trace_queue, suppress_send_errors=True)
                raise asyncio.CancelledError("cancel requested")

            if event_task is not None and event_task in done:
                await ws.send_json({"type": "trace", "event": event_task.result()})
                event_task = None

            if chat_task in done:
                await _drain_trace_queue(ws, trace_queue)
                return chat_task.result()

            if event_task is None:
                event_task = asyncio.create_task(trace_queue.get())
    finally:
        if cancel_watch is not None and not cancel_watch.done():
            cancel_watch.cancel()
        if event_task is not None and not event_task.done():
            event_task.cancel()
        if not chat_task.done():
            chat_task.cancel()
            await asyncio.gather(chat_task, return_exceptions=True)


@app.websocket("/chat")
async def chat_ws(ws: WebSocket):
    session_id = _authenticate_ws(ws)
    if not session_id or not _origin_allowed(ws):
        await ws.close(code=4401)
        return
    await ws.accept()
    logger.info("WS connected session=%s", session_id)

    bus = get_chat_bus()
    bus_queue: asyncio.Queue = asyncio.Queue()
    ws_lock = asyncio.Lock()

    async def _relay(event: dict) -> None:
        await bus_queue.put(event)

    sub_id = await bus.subscribe(session_id, _relay)
    global_sub_id = await bus.subscribe_all(_relay)

    async def _forwarder() -> None:
        while True:
            try:
                event = await bus_queue.get()
                async with ws_lock:
                    await ws.send_json(event)
            except Exception:
                logger.exception("chat bus forwarder error")
                break

    fwd_task = asyncio.create_task(_forwarder())
    history_sent = False

    try:
        graph = get_graph()

        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                if data.get("type") == "desktop_lease":
                    enabled = bool(data.get("enabled"))
                    if enabled:
                        seconds = grant_desktop_lease(session_id)
                        payload = {
                            "type": "desktop_lease",
                            "enabled": True,
                            "seconds": seconds,
                        }
                    else:
                        revoke_desktop_lease(session_id)
                        payload = {"type": "desktop_lease", "enabled": False}
                    async with ws_lock:
                        await ws.send_json(payload)
                    continue
                if data.get("type") == "cancel":
                    request_cancel(session_id)
                    async with ws_lock:
                        await ws.send_json({"type": "cancelled"})
                    continue
                user_text = data.get("text", "").strip()
            except json.JSONDecodeError:
                user_text = raw.strip()

            if not user_text:
                async with ws_lock:
                    await ws.send_json({"type": "error", "text": "Empty message"})
                continue

            if graph and graph._chatdb and not history_sent:
                history_sent = True
                history = graph._chatdb.get_history(session_id, limit=50)
                if history:
                    async with ws_lock:
                        await ws.send_json({"type": "history", "messages": history})

            while not bus_queue.empty():
                try:
                    event = bus_queue.get_nowait()
                    async with ws_lock:
                        await ws.send_json(event)
                except Exception:
                    break

            async with ws_lock:
                await ws.send_json({"type": "thinking"})

            # Run chat as a task so we can race it against incoming WS messages.
            # Only one coroutine reads ws.receive_text() at a time: the outer
            # loop reads when idle; this inner loop reads while chat runs.
            cancel_event = get_cancel_event(session_id)
            chat_task = asyncio.create_task(
                _run_chat_with_trace(ws, user_text, session_id, cancel_event)
            )

            try:
                while not chat_task.done():
                    recv_task = asyncio.create_task(ws.receive_text())
                    done = await _wait_for_first(chat_task, recv_task)
                    if recv_task in done:
                        try:
                            raw_inner = recv_task.result()
                            data = json.loads(raw_inner)
                            if data.get("type") == "cancel":
                                request_cancel(session_id)
                                chat_task.cancel()
                                async with ws_lock:
                                    await ws.send_json({"type": "cancelled"})
                            # Non-cancel messages during chat are consumed and ignored.
                            # The UI should be disabled while busy.
                        except Exception:
                            pass
                    else:
                        recv_task.cancel()

                result = await chat_task
                async with ws_lock:
                    await ws.send_json(_answer_payload(result))
            except asyncio.CancelledError:
                pass  # cancel already sent via cancelled message above
            except asyncio.TimeoutError:
                async with ws_lock:
                    await ws.send_json({"type": "error", "text": "Agent timed out (180s)"})
            except Exception as e:
                logger.exception("chat error")
                async with ws_lock:
                    await ws.send_json({"type": "error", "text": f"Error: {e}"})
            finally:
                if not chat_task.done():
                    chat_task.cancel()
                    try:
                        await chat_task
                    except (asyncio.CancelledError, Exception):
                        pass
    except WebSocketDisconnect:
        logger.info("WS disconnected")
    finally:
        revoke_desktop_lease(session_id)
        await bus.unsubscribe(sub_id)
        await bus.unsubscribe(global_sub_id)
        fwd_task.cancel()
        try:
            await fwd_task
        except asyncio.CancelledError:
            pass


def create_app(graph):
    """Build the FastAPI app with an existing AgentGraph."""
    from agent.shared import set_graph
    set_graph(graph)
    return app


def main():
    import uvicorn
    port = int(os.environ.get("WEB_PORT", "8000"))
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    uvicorn.run("agent.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
