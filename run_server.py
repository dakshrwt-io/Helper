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
logging.getLogger("httpx").setLevel(logging.WARNING)

_REQUIRED_ENV = ("WEB_TOKEN", "DATA_DIR", "LLM_BACKEND", "FILESYSTEM_MCP_DIR")


def require_env(names: tuple[str, ...]) -> None:
    """Fail fast with a clear message when required environment variables are unset.

    Undefined ${VAR} references in config.yaml would otherwise fall through to
    os.path.expandvars and create literal '${VAR}' directories on disk.
    """
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Set them in .env (see .env.example) and restart."
        )


async def main() -> None:
    require_env(_REQUIRED_ENV)

    from agent.graph import AgentGraph
    from agent.shared import set_graph

    # ── 1. Build the agent ────────────────────────────────────────
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    logger.info("Initializing AgentGraph from %s", cfg)
    graph = AgentGraph(config_path=cfg)

    tg_app = None
    try:
        await graph.setup()
        set_graph(graph)
        logger.info(
            "AgentGraph ready — model=%s backend=%s tools=%d",
            graph._model_name,
            graph._llm_backend,
            len(graph.mcp.tool_names) if graph.mcp else 0,
        )

        # ── 2. Telegram bot (if enabled) ──────────────────────────────
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
    finally:
        # ── 4. Cleanup (runs even when setup or serving raises) ───────
        if tg_app:
            logger.info("Shutting down Telegram bot...")
            try:
                await tg_app.updater.stop()
                await tg_app.stop()
                await tg_app.shutdown()
            except Exception:
                logger.exception("Telegram bot shutdown failed")
        logger.info("Shutting down AgentGraph...")
        try:
            await graph.close()
        except Exception:
            logger.exception("AgentGraph shutdown failed")
        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
