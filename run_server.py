"""Launch agent, Telegram bot (optional), and FastAPI web server in one process."""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_server")


async def main() -> None:
    from agent.graph import AgentGraph
    from agent.shared import set_graph

    # ── 1. Build the agent ────────────────────────────────────────
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    logger.info("Initializing AgentGraph from %s", cfg)
    graph = AgentGraph(config_path=cfg)
    await graph.setup()
    set_graph(graph)
    logger.info(
        "AgentGraph ready — model=%s backend=%s tools=%d",
        graph._model_name,
        graph._llm_backend,
        len(graph.mcp.tool_names) if graph.mcp else 0,
    )

    # ── 2. Telegram bot (if enabled) ──────────────────────────────
    tg_app = None
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    enabled = os.environ.get("TELEGRAM_ENABLED", "false").lower() == "true"
    if enabled and token:
        logger.info("Starting Telegram bot")
        from agent.chat_providers.telegram import build_bot

        tg_app = build_bot(token)
        await tg_app.initialize()
        await tg_app.start()
        await tg_app.updater.start_polling(timeout=30)
        logger.info("Telegram bot polling started")
    elif enabled and not token:
        logger.warning("TELEGRAM_ENABLED=true but TELEGRAM_BOT_TOKEN is empty — bot skipped")

    # ── 3. Web server ─────────────────────────────────────────────
    port = int(os.environ.get("WEB_PORT", "8000"))
    host = os.environ.get("WEB_HOST", "127.0.0.1")

    from agent.main import create_app
    import uvicorn

    app = create_app(graph)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    logger.info("Starting web server at http://%s:%d", host, port)
    await server.serve()

    # ── 4. Cleanup ────────────────────────────────────────────────
    logger.info("Shutting down Telegram bot...")
    if tg_app:
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()
    logger.info("Shutting down AgentGraph...")
    await graph.close()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
