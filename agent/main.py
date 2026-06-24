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

from agent.graph import AgentGraph

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent.web")

_graph: AgentGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    _graph = AgentGraph(config_path=cfg)
    await _graph.setup()
    logger.info("AgentGraph ready. tools=%d", len(_graph.mcp.tool_names))
    yield
    await _graph.close()
    logger.info("AgentGraph closed")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "web", "index.html"))


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "tools": len(_graph.mcp.tool_names) if _graph else 0,
        "spent_today": _graph._chatdb.spent_today() if _graph and _graph._chatdb else 0.0,
        "daily_cap": _graph._daily_cap if _graph else 0.0,
    })


async def _run_chat_with_trace(ws: WebSocket, user_text: str, session_id: str) -> dict:
    """Run a turn while forwarding its internal execution events to the client."""
    trace_queue: asyncio.Queue[dict] = asyncio.Queue()
    chat_task = asyncio.create_task(
        _graph.chat(user_text, session_id=session_id, trace_queue=trace_queue)
    )
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
    try:
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
                await ws.send_json({"type": "error", "text": "Empty message"})
                continue

            await ws.send_json({"type": "thinking"})
            try:
                result = await _run_chat_with_trace(ws, user_text, session_id)
                await ws.send_json({
                    "type": "answer",
                    "text": result["text"],
                    "cost_spent": round(result["cost_spent"], 6),
                    "spent_today": round(_graph._chatdb.spent_today(), 6) if _graph._chatdb else 0.0,
                    "daily_cap": _graph._daily_cap,
                })
            except asyncio.TimeoutError:
                await ws.send_json({"type": "error", "text": "Agent timed out (180s)"})
            except Exception as e:
                logger.exception("chat error")
                await ws.send_json({"type": "error", "text": f"Error: {e}"})
    except WebSocketDisconnect:
        logger.info("WS disconnected")


def main():
    import uvicorn
    port = int(os.environ.get("WEB_PORT", "8000"))
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    uvicorn.run("agent.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
