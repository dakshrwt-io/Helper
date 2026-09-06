"""LangGraph agent: load_memory -> agent -> tools -> save_memory -> END."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict

from agent.agent_helpers import extract_final_ai_text
from agent.config_loader import load_agent_config
from agent.mcp_adapter import build_langchain_tools
from agent.mcp_manager import MCPManager
from agent.memory.db import ChatDB
from agent.memory_manager import MemoryManager
from agent.memory.vector import VectorStore
from agent.react import build_react_graph
from agent.subagents import SubAgentManager, build_task_tool
from agent.tools.computer import (
    clear_screenshots,
    get_computer_tools,
)
from agent.shared import activate_session
from agent.trace import TraceCollector, activate_trace, duration_ms

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    messages: list[Any]
    session_id: str
    tool_calls_made: int
    llm_calls_made: int
    trace: TraceCollector
    started_at: float
    stopped_reason: str  # "max_iterations" | "max_seconds" | "cancelled" | ""
    cancel_event: Any  # asyncio.Event — cooperative cancellation token


class AgentGraph:
    def __init__(
        self,
        config_path: str = "config.yaml",
        mcp_manager: MCPManager | None = None,
    ) -> None:
        self.config_path = config_path
        self.mcp = mcp_manager or MCPManager(config_path)
        self._llm = None
        self._llm_with_tools = None
        self._tools: list[Any] = []
        self._graph = None
        self._persona: str = ""
        self._cfg: dict[str, Any] = {}
        self._chatdb: ChatDB | None = None
        self._vector: VectorStore | None = None
        self._llm_backend: str = "openrouter"
        self._model_name: str = ""
        self._vision_llm: Any = None
        self._vision_model: str = ""
        self._subagent_manager: SubAgentManager | None = None
        self._max_history_tokens: int = 4000
        self._memory_manager: MemoryManager | None = None

    def _make_llm(self) -> None:
        """Build the LLM based on the configured backend."""
        from agent.llm_factory import build_llm

        llm_cfg = self._cfg.get("llm", {})
        self._llm_backend = llm_cfg.get("backend", "openrouter").lower()
        temperature = self._cfg.get("agent", {}).get("temperature", 0.3)
        max_retries = int(llm_cfg.get("max_retries", 3))
        timeout = int(llm_cfg.get("timeout_seconds", 60))

        self._llm, self._model_name = build_llm(
            llm_cfg=llm_cfg,
            backend=self._llm_backend,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )

        vision_cfg = llm_cfg.get("vision", {})
        vision_key = vision_cfg.get("api_key", "")
        # Fall back to groq config when vision section is unconfigured
        if (not vision_key or vision_key in ("", "REPLACE_ME")) and llm_cfg.get("groq", {}).get("api_key", ""):
            vision_cfg = llm_cfg.get("groq", {})
            vision_key = vision_cfg.get("api_key", "")
        if vision_key and vision_key not in ("", "REPLACE_ME"):
            from langchain_openai import ChatOpenAI

            self._vision_model = vision_cfg.get("model", "google/gemini-2.5-flash")
            self._vision_llm = ChatOpenAI(
                model=self._vision_model,
                openai_api_key=vision_key,
                openai_api_base=vision_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                temperature=0.0,
                timeout=timeout,
                max_retries=max_retries,
            )
            logger.info("Vision LLM ready: %s", self._vision_model)

    async def setup(self) -> None:
        """Async init: load config, start MCP, build LLM+tools+graph."""
        setup_start = time.perf_counter()
        phase_start = time.perf_counter()
        loaded_config = load_agent_config(self.config_path)
        self._cfg = loaded_config.config
        self._persona = loaded_config.persona
        self._chatdb = loaded_config.chatdb
        self._vector = loaded_config.vector
        self._max_history_tokens = loaded_config.max_history_tokens
        logger.info("Agent setup load completed in %d ms", duration_ms(phase_start))

        phase_start = time.perf_counter()
        await self.mcp.start()
        logger.info("Agent setup MCP start completed in %d ms", duration_ms(phase_start))

        phase_start = time.perf_counter()
        self._make_llm()
        logger.info("Agent setup LLM init completed in %d ms", duration_ms(phase_start))

        self._ensure_memory_manager()

        phase_start = time.perf_counter()
        tools = await build_langchain_tools(self.mcp)
        logger.info(
            "Agent setup tool wrapping completed in %d ms",
            duration_ms(phase_start),
        )

        phase_start = time.perf_counter()
        cc = self._cfg.get("computer_control", {})
        if str(cc.get("enabled", "")).lower() == "true":
            tools.extend(get_computer_tools())
        logger.info(
            "Agent setup computer tools completed in %d ms",
            duration_ms(phase_start),
        )

        phase_start = time.perf_counter()
        if self._cfg.get("subagents", {}).get("enabled", True):
            self._subagent_manager = SubAgentManager(
                raw_config=self._cfg,
                mcp_tools=[t for t in tools if not t.name.startswith("computer_")],
                computer_tools=[t for t in tools if t.name.startswith("computer_")],
                llm=self._llm,
                llm_backend=self._llm_backend,
                model_name=self._model_name,
                vision_llm=self._vision_llm,
                vision_model=self._vision_model,
            )
            if self._subagent_manager.agent_names:
                tools.append(build_task_tool(self._subagent_manager))
        logger.info(
            "Agent setup subagents completed in %d ms",
            duration_ms(phase_start),
        )

        phase_start = time.perf_counter()
        self._tools = tools
        self._llm_with_tools = self._llm.bind_tools(tools) if tools else self._llm
        self._build_graph()
        logger.info(
            "Agent setup graph compile completed in %d ms",
            duration_ms(phase_start),
        )
        logger.info("Agent setup completed in %d ms", duration_ms(setup_start))

    def _current_system_prompt(self) -> str:
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"{self._persona}\n\nCurrent datetime: {current_time}"

    def _ensure_memory_manager(self) -> MemoryManager:
        """Create or refresh the memory manager with current stores and LLM."""
        if self._memory_manager is None:
            self._memory_manager = MemoryManager(
                self._chatdb,
                self._vector,
                self._llm,
                self._max_history_tokens,
            )
        else:
            self._memory_manager.chatdb = self._chatdb
            self._memory_manager.vector = self._vector
            self._memory_manager.llm = self._llm
            self._memory_manager.max_history_tokens = self._max_history_tokens
        return self._memory_manager

    def _build_graph(self) -> None:
        agent_cfg = self._cfg.get("agent", {})
        self._graph = build_react_graph(
            AgentState,
            self._llm_with_tools,
            self._tools,
            max_iter=int(agent_cfg.get("max_iterations", 15)),
            max_secs=float(agent_cfg.get("max_seconds", 120)),
            system_prompt=self._current_system_prompt,
            model_name=self._model_name,
            backend=self._llm_backend,
            vision_llm=self._vision_llm,
            vision_model=self._vision_model,
        )

    _STOP_REASONS = {
        "max_iterations": "iteration limit",
        "max_seconds": "time limit",
        "cancelled": "user request",
    }

    @staticmethod
    def _extract_final_text(
        out_msgs: list[Any], stopped_reason: str = ""
    ) -> str:
        final = extract_final_ai_text(out_msgs)
        if final:
            if final.startswith("[Thinking]"):
                logger.warning(
                    "Reasoning content leaked into final answer — "
                    "this indicates a bug in reasoning-content handling"
                )
            if stopped_reason:
                # Build progress summary from tool calls
                tools_used: list[str] = []
                for m in out_msgs:
                    if isinstance(m, ToolMessage):
                        tools_used.append(getattr(m, "name", "tool"))
                unique_tools = list(dict.fromkeys(tools_used))  # dedup, preserve order
                tool_list = ", ".join(unique_tools[:8]) if unique_tools else "none"
                reason = AgentGraph._STOP_REASONS.get(stopped_reason, stopped_reason)
                final = (
                    f"[Stopped by {reason} after {len(tools_used)} actions]\n"
                    f"Tools used: {tool_list}.\n"
                    f"Last state: {final}"
                )
            return final
        if stopped_reason:
            reason = AgentGraph._STOP_REASONS.get(stopped_reason, stopped_reason)
            return (
                f"I hit the {reason} before completing this task. "
                "Please try a simpler request or continue from where I left off."
            )
        return (
            "I wasn't able to complete this task within the iteration limit. "
            "Please try a simpler request or rephrase."
        )

    async def chat(
        self,
        user_text: str,
        session_id: str = "default",
        trace_queue: asyncio.Queue[dict[str, Any]] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        """Run one turn and emit its LLM and tool execution trace."""
        from agent.shared import clear_cancel

        trace = TraceCollector(queue=trace_queue)
        turn_start = time.perf_counter()

        try:
            with activate_session(session_id), activate_trace(trace):
                trace.emit("turn_started", session_id=session_id)
                clear_screenshots()
                try:
                    memory = self._ensure_memory_manager()
                    prior, summary = await memory._load_history_messages(
                        session_id,
                        trace,
                    )
                    # Compute turn hashes from prior to deduplicate vector recall
                    prior_hashes: set[str] = set()
                    for i in range(0, len(prior) - 1, 2):
                        if isinstance(prior[i], HumanMessage) and isinstance(prior[i + 1], AIMessage):
                            prior_hashes.add(str(hash(str(prior[i].content) + "|" + str(prior[i + 1].content))))
                    recalled = await memory._recall_context(
                        user_text,
                        session_id,
                        trace,
                        prior_hashes,
                    )
                    msgs = memory._build_turn_messages(
                        user_text,
                        recalled,
                        prior,
                        summary,
                    )
                    state: AgentState = {
                        "messages": msgs,
                        "session_id": session_id,
                        "tool_calls_made": 0,
                        "llm_calls_made": 0,
                        "trace": trace,
                        "started_at": time.perf_counter(),
                        "cancel_event": cancel_event,
                    }
                    result = await self._graph.ainvoke(state)
                    out_msgs = result.get("messages", [])
                    final = self._extract_final_text(out_msgs, result.get("stopped_reason", ""))

                    await memory._persist_turn(
                        session_id,
                        user_text,
                        final,
                        trace,
                    )
                    trace.emit(
                        "turn_completed",
                        duration_ms=duration_ms(turn_start),
                        llm_calls=result.get("llm_calls_made", 0),
                        tool_rounds=result.get("tool_calls_made", 0),
                    )
                    return {
                        "text": final,
                        "messages": out_msgs,
                        "trace": trace.events,
                    }
                except Exception as exc:
                    trace.emit(
                        "turn_failed",
                        duration_ms=duration_ms(turn_start),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
        finally:
            clear_cancel(session_id)

    async def close(self) -> None:
        await self.mcp.stop()
