"""LangGraph agent: load_memory -> agent -> tools -> save_memory -> END."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.agent_helpers import analyze_screenshot, extract_final_ai_text
from agent.config_loader import load_agent_config
from agent.mcp_adapter import build_langchain_tools
from agent.mcp_manager import MCPManager
from agent.memory.db import ChatDB
from agent.memory_manager import MemoryManager
from agent.memory.vector import VectorStore
from agent.subagents import SubAgentManager, build_task_tool
from agent.tools.computer import (
    clear_screenshots,
    get_computer_tools,
)
from agent.shared import activate_session, sanitize_aimessage
from agent.trace import TraceCollector, activate_trace, display_content, duration_ms

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
        self._tools_node: ToolNode | None = None
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
                tools.append(build_task_tool(self._subagent_manager, self._chatdb))
        logger.info(
            "Agent setup subagents completed in %d ms",
            duration_ms(phase_start),
        )

        phase_start = time.perf_counter()
        self._llm_with_tools = self._llm.bind_tools(tools) if tools else self._llm
        self._tools_node = ToolNode(tools) if tools else None
        self._build_graph()
        logger.info(
            "Agent setup graph compile completed in %d ms",
            duration_ms(phase_start),
        )
        logger.info("Agent setup completed in %d ms", duration_ms(setup_start))

    def _build_graph(self) -> None:
        max_iter = int(self._cfg.get("agent", {}).get("max_iterations", 15))
        max_secs = float(self._cfg.get("agent", {}).get("max_seconds", 120))

        async def agent_node(state: AgentState) -> dict[str, Any]:
            cancel = state.get("cancel_event")
            if cancel and cancel.is_set():
                return {
                    "messages": state.get("messages", []),
                    "stopped_reason": "cancelled",
                }
            msgs = list(state.get("messages", []))
            if not msgs or not isinstance(msgs[0], SystemMessage):
                current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                persona_with_time = f"{self._persona}\n\nCurrent datetime: {current_time}"
                msgs = [SystemMessage(content=persona_with_time)] + msgs
            msgs = [sanitize_aimessage(m) if isinstance(m, AIMessage) else m for m in msgs]
            trace = state.get("trace")
            llm_call = state.get("llm_calls_made", 0) + 1

            if self._vision_llm is not None and msgs and isinstance(msgs[-1], ToolMessage) and getattr(msgs[-1], "name", None) == "computer_screenshot":
                msgs = await analyze_screenshot(
                    msgs,
                    self._vision_llm,
                    self._vision_model,
                    trace,
                    log=logger,
                    failure_message="Vision call failed: %s",
                )

            start = time.perf_counter()
            if trace:
                trace.emit(
                    "llm_started",
                    llm_call=llm_call,
                    model=self._model_name,
                    backend=self._llm_backend,
                    message_count=len(msgs),
                )
            try:
                resp = await self._llm_with_tools.ainvoke(msgs)
                cancel = state.get("cancel_event")
                if cancel and cancel.is_set():
                    return {
                        "messages": state.get("messages", []),
                        "stopped_reason": "cancelled",
                    }
            except Exception as exc:
                if trace:
                    trace.emit(
                        "llm_failed",
                        llm_call=llm_call,
                        duration_ms=duration_ms(start),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                raise
            reasoning = getattr(resp, "additional_kwargs", {}) or {}
            reasoning_text = reasoning.get("reasoning_content", "")
            if reasoning_text and trace:
                trace.emit("reasoning", content=reasoning_text)
            tc = state.get("tool_calls_made", 0)
            usage = getattr(resp, "usage_metadata", None)
            if trace:
                trace.emit(
                    "llm_completed",
                    llm_call=llm_call,
                    duration_ms=duration_ms(start),
                    content=display_content(resp.content),
                    tool_calls=getattr(resp, "tool_calls", None) or [],
                    usage=usage or {},
                )
            # Detect forced stop: LLM wants more tool calls but limits prevent next iteration
            has_tool_calls = bool(getattr(resp, "tool_calls", None))
            stopped_reason = ""
            if has_tool_calls:
                if tc + 1 >= max_iter:
                    stopped_reason = "max_iterations"
                elif max_secs > 0 and state.get("started_at"):
                    if time.perf_counter() - state["started_at"] >= max_secs:
                        stopped_reason = "max_seconds"
            result: dict[str, Any] = {
                "messages": msgs + [resp],
                "tool_calls_made": tc,
                "llm_calls_made": llm_call,
            }
            if stopped_reason:
                result["stopped_reason"] = stopped_reason
            return result

        async def tools_node(state: AgentState) -> dict[str, Any]:
            cancel = state.get("cancel_event")
            if cancel and cancel.is_set():
                return {"messages": state.get("messages", [])}
            if self._tools_node is None:
                return {}
            with activate_trace(state.get("trace")):
                out = await self._tools_node.ainvoke(state)
            new_msgs = out.get("messages", [])
            msgs = list(state.get("messages", [])) + new_msgs
            return {"messages": msgs, "tool_calls_made": state.get("tool_calls_made", 0) + 1}

        def route(state: AgentState) -> str:
            if state.get("stopped_reason") == "cancelled":
                return END
            if state.get("tool_calls_made", 0) >= max_iter:
                return END
            if max_secs > 0 and state.get("started_at"):
                elapsed = time.perf_counter() - state["started_at"]
                if elapsed >= max_secs:
                    return END
            msgs = state.get("messages", [])
            if not msgs:
                return END
            last = msgs[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        g = StateGraph(AgentState)
        g.add_node("agent", agent_node)
        g.add_node("tools", tools_node)
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        self._graph = g.compile()

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
                        name = getattr(m, "name", "tool")
                        tools_used.append(name)
                unique_tools = list(dict.fromkeys(tools_used))  # dedup, preserve order
                tool_list = ", ".join(unique_tools[:8]) if unique_tools else "none"
                reason = {"max_iterations": "iteration limit", "max_seconds": "time limit", "cancelled": "user request"}.get(
                    stopped_reason, stopped_reason
                )
                final = (
                    f"[Stopped by {reason} after {len(tools_used)} actions]\n"
                    f"Tools used: {tool_list}.\n"
                    f"Last state: {final}"
                )
            return final
        if stopped_reason:
            reason = {"max_iterations": "iteration limit", "max_seconds": "time limit"}.get(
                stopped_reason, stopped_reason
            )
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
                    memory = self._memory_manager
                    if memory is None:
                        memory = MemoryManager(
                            self._chatdb,
                            self._vector,
                            self._llm,
                            self._max_history_tokens,
                        )
                        self._memory_manager = memory
                    else:
                        memory.chatdb = self._chatdb
                        memory.vector = self._vector
                        memory.llm = self._llm
                        memory.max_history_tokens = self._max_history_tokens
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
