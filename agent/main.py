"""FastAPI server: WebSocket /chat + static HTML frontend."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from agent.shared import get_graph, get_chat_lock, get_chat_bus

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent.web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph = get_graph()
    if graph:
        logger.info("AgentGraph ready. tools=%d", len(graph.mcp.tool_names) if graph.mcp else 0)
    yield
    logger.info("Web server shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "web", "index.html"))


@app.get("/health")
async def health():
    graph = get_graph()
    return JSONResponse({
        "status": "ok",
        "tools": len(graph.mcp.tool_names) if graph and graph.mcp else 0,
    })


async def _run_chat_with_trace(ws: WebSocket, user_text: str, session_id: str) -> dict:
    """Run a turn while forwarding its internal execution events to the client."""
    graph = get_graph()
    trace_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def _locked_chat():
        async with get_chat_lock():
            return await graph.chat(user_text, session_id=session_id, trace_queue=trace_queue)

    chat_task = asyncio.create_task(_locked_chat())
    event_task: asyncio.Task | None = asyncio.create_task(trace_queue.get())
    deadline = asyncio.get_running_loop().time() + 180

    try:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            waiting = {chat_task}
            if event_task is not None:
                waiting.add(event_task)
            done, _ = await asyncio.wait(
                waiting,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise asyncio.TimeoutError

            if event_task is not None and event_task in done:
                await ws.send_json({"type": "trace", "event": event_task.result()})
                event_task = None

            if chat_task in done:
                while not trace_queue.empty():
                    await ws.send_json({"type": "trace", "event": trace_queue.get_nowait()})
                return chat_task.result()

            if event_task is None:
                event_task = asyncio.create_task(trace_queue.get())
    finally:
        if event_task is not None and not event_task.done():
            event_task.cancel()
        if not chat_task.done():
            chat_task.cancel()
            await asyncio.gather(chat_task, return_exceptions=True)


@app.websocket("/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    logger.info("WS connected")

    bus = get_chat_bus()
    bus_queue: asyncio.Queue = asyncio.Queue()
    ws_lock = asyncio.Lock()

    async def _relay(event: dict) -> None:
        await bus_queue.put(event)

    sub_id = await bus.subscribe(_relay)

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
    _history_sent: dict[str, bool] = {}

    try:
        graph = get_graph()

        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                user_text = data.get("text", "").strip()
                session_id = data.get("session_id", "default")
            except json.JSONDecodeError:
                user_text = raw.strip()
                session_id = "default"

            if not user_text:
                async with ws_lock:
                    await ws.send_json({"type": "error", "text": "Empty message"})
                continue

            if graph and graph._chatdb and not _history_sent.get(session_id):
                _history_sent[session_id] = True
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

            try:
                result = await _run_chat_with_trace(ws, user_text, session_id)
                async with ws_lock:
                    await ws.send_json({
                        "type": "answer",
                        "text": result["text"],
                    })
            except asyncio.TimeoutError:
                async with ws_lock:
                    await ws.send_json({"type": "error", "text": "Agent timed out (180s)"})
            except Exception as e:
                logger.exception("chat error")
                async with ws_lock:
                    await ws.send_json({"type": "error", "text": f"Error: {e}"})
    except WebSocketDisconnect:
        logger.info("WS disconnected")
    finally:
        await bus.unsubscribe(sub_id)
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
