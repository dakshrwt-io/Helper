"""LangGraph agent: load_memory -> agent -> tools -> save_memory -> END."""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.mcp_adapter import build_langchain_tools
from agent.mcp_manager import MCPManager
from agent.memory.db import ChatDB
from agent.memory.vector import VectorStore
from agent.subagents import SubAgentManager, build_task_tool
from agent.tools.computer import (
    clear_screenshots,
    get_computer_tools,
    inject_screenshots,
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

    def _load(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def _expand(obj: Any) -> Any:
            if isinstance(obj, str):
                return os.path.expandvars(obj)
            if isinstance(obj, dict):
                return {k: _expand(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_expand(v) for v in obj]
            return obj

        self._cfg = _expand(raw)
        ag = self._cfg.get("agent", {})
        p = os.path.expandvars(ag.get("persona_path", "agent/persona.md"))
        try:
            with open(p, "r", encoding="utf-8") as f:
                self._persona = f.read()
        except OSError:
            self._persona = "You are a helpful personal AI assistant."
        mem = self._cfg.get("memory", {})
        self._chatdb = ChatDB(os.path.expandvars(mem.get("sqlite", "data/history.db")))
        self._vector = VectorStore(os.path.expandvars(mem.get("chroma", "data/chroma")))

        if self._vector.count() == 0:
            turns = self._chatdb.export_all_turns()
            if turns:
                logger.info(
                    "Vector store is empty — rebuilding from %d SQLite turns",
                    len(turns),
                )
                self._vector.rebuild(
                    ids=[str(uuid.uuid4()) for _ in turns],
                    texts=[t["text"] for t in turns],
                    metas=[{"session_id": t["session_id"]} for t in turns],
                )

    def _make_llm(self) -> None:
        """Build the LLM based on the configured backend."""
        from langchain_openai import ChatOpenAI

        llm_cfg = self._cfg.get("llm", {})
        self._llm_backend = llm_cfg.get("backend", "openrouter").lower()
        temperature = self._cfg.get("agent", {}).get("temperature", 0.3)

        if self._llm_backend == "ollama":
            from langchain_ollama import ChatOllama

            ocfg = llm_cfg.get("ollama", {})
            self._model_name = ocfg.get("model", "gemma4:e2b")
            self._llm = ChatOllama(
                model=self._model_name,
                base_url=ocfg.get("base_url", "http://127.0.0.1:11434"),
                temperature=temperature,
            )
        elif self._llm_backend == "deepseek":
            ocfg = llm_cfg.get("deepseek", {})
            self._model_name = ocfg.get("model", "deepseek-v4-flash")
            model_kwargs: dict[str, Any] = {}
            extra_body: dict[str, Any] = {}
            if str(ocfg.get("thinking", "")).lower() == "true":
                extra_body["thinking"] = {"type": "enabled"}
            reasoning = ocfg.get("reasoning_effort", "")
            if reasoning:
                extra_body["reasoning_effort"] = reasoning
            if extra_body:
                model_kwargs["extra_body"] = extra_body
            self._llm = ChatOpenAI(
                model=self._model_name,
                openai_api_key=ocfg.get("api_key", ""),
                openai_api_base=ocfg.get("base_url", "https://api.deepseek.com"),
                temperature=temperature,
                model_kwargs=model_kwargs if model_kwargs else None,
                timeout=60,
                max_retries=1,
            )
        elif self._llm_backend == "nvidia":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            ocfg = llm_cfg.get("nvidia", {})
            self._model_name = ocfg.get("model", "minimaxai/minimax-m3")
            self._llm = ChatNVIDIA(
                model=self._model_name,
                api_key=ocfg.get("api_key", ""),
                temperature=float(ocfg.get("temperature", temperature)),
                top_p=float(ocfg.get("top_p", 0.95)),
                max_completion_tokens=int(ocfg.get("max_completion_tokens", 8192)),
            )
        else:
            ocfg = llm_cfg.get("openrouter", {})
            self._model_name = ocfg.get("model", "z-ai/glm-4.5")
            self._llm = ChatOpenAI(
                model=self._model_name,
                openai_api_key=ocfg.get("api_key", ""),
                openai_api_base=ocfg.get("base_url", "https://openrouter.ai/api/v1"),
                temperature=temperature,
                timeout=60,
                max_retries=1,
            )

        vision_cfg = llm_cfg.get("vision", {})
        vision_key = vision_cfg.get("api_key", "")
        if vision_key and vision_key not in ("", "REPLACE_ME"):
            self._vision_model = vision_cfg.get("model", "google/gemini-2.5-flash")
            self._vision_llm = ChatOpenAI(
                model=self._vision_model,
                openai_api_key=vision_key,
                openai_api_base=vision_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                temperature=0.0,
                timeout=60,
                max_retries=1,
            )
            logger.info("Vision LLM ready: %s", self._vision_model)

    async def setup(self) -> None:
        """Async init: load config, start MCP, build LLM+tools+graph."""
        setup_start = time.perf_counter()
        phase_start = time.perf_counter()
        self._load()
        logger.info("Agent setup load completed in %d ms", duration_ms(phase_start))

        phase_start = time.perf_counter()
        await self.mcp.start()
        logger.info("Agent setup MCP start completed in %d ms", duration_ms(phase_start))

        phase_start = time.perf_counter()
        self._make_llm()
        logger.info("Agent setup LLM init completed in %d ms", duration_ms(phase_start))

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
            msgs = list(state.get("messages", []))
            if not msgs or not isinstance(msgs[0], SystemMessage):
                current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                persona_with_time = f"{self._persona}\n\nCurrent datetime: {current_time}"
                msgs = [SystemMessage(content=persona_with_time)] + msgs
            msgs = [sanitize_aimessage(m) if isinstance(m, AIMessage) else m for m in msgs]
            trace = state.get("trace")
            llm_call = state.get("llm_calls_made", 0) + 1

            if self._vision_llm is not None and msgs and isinstance(msgs[-1], ToolMessage) and getattr(msgs[-1], "name", None) == "computer_screenshot":
                vision_start = time.perf_counter()
                if trace:
                    trace.emit(
                        "vision_started",
                        model=self._vision_model,
                        backend="vision",
                    )
                try:
                    vision_msgs = inject_screenshots(msgs)
                    vision_resp = await self._vision_llm.ainvoke(vision_msgs)
                    vision_text = display_content(vision_resp.content)
                    msgs[-1] = msgs[-1].model_copy(
                        update={"content": f"[Screen analysis]\n{vision_text}"}
                    )
                    if trace:
                        trace.emit(
                            "vision_completed",
                            duration_ms=duration_ms(vision_start),
                            content=display_content(vision_resp.content),
                            usage=getattr(vision_resp, "usage_metadata", None) or {},
                        )
                except Exception as exc:
                    logger.warning("Vision call failed: %s", exc)
                    if trace:
                        trace.emit(
                            "vision_failed",
                            duration_ms=duration_ms(vision_start),
                            error_type=type(exc).__name__,
                            error=str(exc),
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
            if reasoning_text:
                if trace:
                    trace.emit("reasoning", content=reasoning_text)
                resp = resp.model_copy(update={"content": f"[Thinking]\n{reasoning_text}\n\n{resp.content}"})
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
            return {
                "messages": msgs + [resp],
                "tool_calls_made": tc,
                "llm_calls_made": llm_call,
            }

        async def tools_node(state: AgentState) -> dict[str, Any]:
            if self._tools_node is None:
                return {}
            with activate_trace(state.get("trace")):
                out = await self._tools_node.ainvoke(state)
            new_msgs = out.get("messages", [])
            msgs = list(state.get("messages", [])) + new_msgs
            return {"messages": msgs, "tool_calls_made": state.get("tool_calls_made", 0) + 1}

        def route(state: AgentState) -> str:
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

    async def _recall_context(
        self, user_text: str, session_id: str, trace: TraceCollector
    ) -> list[str]:
        if not self._vector:
            return []

        recall_start = time.perf_counter()
        trace.emit("memory_recall_started")
        hits = await asyncio.to_thread(
            self._vector.query, user_text, 3, session_id
        )
        recalled = [h["text"] for h in hits if h.get("distance", 1.0) < 0.6]
        trace.emit(
            "memory_recall_completed",
            duration_ms=duration_ms(recall_start),
            matches=len(recalled),
        )
        return recalled

    async def _load_history_messages(self, session_id: str, trace: TraceCollector) -> list[Any]:
        if not self._chatdb:
            return []

        history_start = time.perf_counter()
        prior: list[Any] = []
        history = await asyncio.to_thread(self._chatdb.get_history, session_id, 20)
        for m in history:
            if m["role"] == "user":
                prior.append(HumanMessage(content=m["content"]))
            else:
                prior.append(AIMessage(content=m["content"]))
        trace.emit(
            "memory_history_loaded",
            duration_ms=duration_ms(history_start),
            messages=len(prior),
        )
        return prior

    def _build_turn_messages(
        self,
        user_text: str,
        recalled: list[str],
        prior: list[Any],
    ) -> list[Any]:
        msgs: list[Any] = []
        if recalled:
            msgs.append(
                SystemMessage(
                    content="\n\nRelevant past context:\n" + "\n---\n".join(recalled)
                )
            )
        msgs.extend(prior)
        msgs.append(HumanMessage(content=user_text))
        return msgs

    @staticmethod
    def _extract_final_text(out_msgs: list[Any]) -> str:
        final = ""
        for m in reversed(out_msgs):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                final = display_content(m.content)
                break
        if not final:
            for m in reversed(out_msgs):
                if isinstance(m, AIMessage):
                    final = display_content(m.content) if m.content else ""
                    break
        if final:
            return final
        return (
            "I wasn't able to complete this task within the iteration limit. "
            "Please try a simpler request or rephrase."
        )

    async def _persist_turn(
        self,
        session_id: str,
        user_text: str,
        final: str,
        trace: TraceCollector,
    ) -> None:
        persist_start = time.perf_counter()
        if self._chatdb:
            add_turn = getattr(self._chatdb, "add_turn", None)
            if callable(add_turn):
                await asyncio.to_thread(add_turn, session_id, user_text, final)
            else:
                await asyncio.to_thread(
                    self._chatdb.add_message,
                    session_id,
                    "user",
                    user_text,
                )
                await asyncio.to_thread(
                    self._chatdb.add_message,
                    session_id,
                    "assistant",
                    final,
                )
        if self._vector and final:
            await asyncio.to_thread(
                self._vector.add,
                [str(uuid.uuid4())],
                [f"User: {user_text}\nAssistant: {final}"],
                [{"session_id": session_id}],
            )
        trace.emit("memory_persisted", duration_ms=duration_ms(persist_start))

    async def chat(
        self,
        user_text: str,
        session_id: str = "default",
        trace_queue: asyncio.Queue[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run one turn and emit its LLM and tool execution trace."""
        trace = TraceCollector(queue=trace_queue)
        turn_start = time.perf_counter()

        with activate_session(session_id), activate_trace(trace):
            trace.emit("turn_started", session_id=session_id)
            clear_screenshots()
            try:
                recalled = await self._recall_context(user_text, session_id, trace)
                prior = await self._load_history_messages(session_id, trace)
                msgs = self._build_turn_messages(user_text, recalled, prior)
                state: AgentState = {
                    "messages": msgs,
                    "session_id": session_id,
                    "tool_calls_made": 0,
                    "llm_calls_made": 0,
                    "trace": trace,
                    "started_at": time.perf_counter(),
                }
                result = await self._graph.ainvoke(state)
                out_msgs = result.get("messages", [])
                final = self._extract_final_text(out_msgs)

                await self._persist_turn(session_id, user_text, final, trace)
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

    async def close(self) -> None:
        await self.mcp.stop()
