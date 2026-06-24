"""LangGraph agent: load_memory -> agent -> tools -> save_memory -> END."""
from __future__ import annotations

import asyncio
import logging
import os
import time
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
from agent.trace import TraceCollector, activate_trace, display_content, duration_ms

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    messages: list[Any]
    cost_spent: float
    session_id: str
    tool_calls_made: int
    llm_calls_made: int
    trace: TraceCollector


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
        self._daily_cap: float = 1.0
        self._chatdb: ChatDB | None = None
        self._vector: VectorStore | None = None
        self._llm_backend: str = "openrouter"
        self._model_name: str = ""

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
        self._daily_cap = float(self._cfg.get("daily_cost_usd", 1.0))
        mem = self._cfg.get("memory", {})
        self._chatdb = ChatDB(os.path.expandvars(mem.get("sqlite", "data/history.db")))
        self._vector = VectorStore(os.path.expandvars(mem.get("chroma", "data/chroma")))

    def _make_llm(self) -> None:
        """Build the LLM based on the configured backend."""
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
            from langchain_openai import ChatOpenAI

            ocfg = llm_cfg.get("deepseek", {})
            self._model_name = ocfg.get("model", "deepseek-v4-flash")
            self._llm = ChatOpenAI(
                model=self._model_name,
                openai_api_key=ocfg.get("api_key", ""),
                openai_api_base=ocfg.get("base_url", "https://api.deepseek.com"),
                temperature=temperature,
            )
        else:
            from langchain_openai import ChatOpenAI

            ocfg = llm_cfg.get("openrouter", {})
            self._model_name = ocfg.get("model", "z-ai/glm-4.5")
            self._llm = ChatOpenAI(
                model=self._model_name,
                openai_api_key=ocfg.get("api_key", ""),
                openai_api_base=ocfg.get("base_url", "https://openrouter.ai/api/v1"),
                temperature=temperature,
            )

    async def setup(self) -> None:
        """Async init: load config, start MCP, build LLM+tools+graph."""
        self._load()
        await self.mcp.start()
        tools = await build_langchain_tools(self.mcp)
        self._make_llm()
        self._llm_with_tools = self._llm.bind_tools(tools) if tools else self._llm
        self._tools_node = ToolNode(tools) if tools else None
        self._build_graph()

    def _build_graph(self) -> None:
        max_iter = int(self._cfg.get("agent", {}).get("max_iterations", 15))

        async def agent_node(state: AgentState) -> dict[str, Any]:
            msgs = list(state.get("messages", []))
            if not msgs or not isinstance(msgs[0], SystemMessage):
                msgs = [SystemMessage(content=self._persona)] + msgs
            trace = state.get("trace")
            llm_call = state.get("llm_calls_made", 0) + 1
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
            cost = state.get("cost_spent", 0.0)
            tc = state.get("tool_calls_made", 0)
            if self._llm_backend != "ollama":
                usage = getattr(resp, "usage_metadata", None)
                if usage:
                    total = usage.get("total_tokens") or (
                        usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    )
                    cost += float(total) * 0.000002
            else:
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
                "cost_spent": cost,
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
            if state.get("cost_spent", 0.0) >= self._daily_cap:
                return END
            if state.get("tool_calls_made", 0) >= max_iter:
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

    async def chat(
        self,
        user_text: str,
        session_id: str = "default",
        trace_queue: asyncio.Queue[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run one turn and emit its LLM and tool execution trace."""
        trace = TraceCollector(queue=trace_queue)
        turn_start = time.perf_counter()

        with activate_trace(trace):
            trace.emit("turn_started", session_id=session_id)
            try:
                # 0. cost guard: hard stop if daily cap already hit
                if self._chatdb and self._chatdb.spent_today() >= self._daily_cap:
                    text = f"Daily cost cap (${self._daily_cap:.2f}) reached. Try again after UTC midnight."
                    trace.emit("turn_blocked", reason="daily_cost_cap", daily_cap=self._daily_cap)
                    trace.emit("turn_completed", duration_ms=duration_ms(turn_start), cost_spent=0.0)
                    return {
                        "text": text,
                        "cost_spent": 0.0,
                        "messages": [],
                        "trace": trace.events,
                    }

                # 1. recall: fetch relevant past turns
                recalled: list[str] = []
                if self._vector:
                    recall_start = time.perf_counter()
                    trace.emit("memory_recall_started")
                    hits = self._vector.query(user_text, top_k=3)
                    recalled = [h["text"] for h in hits if h.get("distance", 1.0) < 0.6]
                    trace.emit(
                        "memory_recall_completed",
                        duration_ms=duration_ms(recall_start),
                        matches=len(recalled),
                    )

                # 2. load SQLite history for this session
                prior: list[Any] = []
                if self._chatdb:
                    history_start = time.perf_counter()
                    for m in self._chatdb.get_history(session_id, limit=20):
                        if m["role"] == "user":
                            prior.append(HumanMessage(content=m["content"]))
                        else:
                            prior.append(AIMessage(content=m["content"]))
                    trace.emit(
                        "memory_history_loaded",
                        duration_ms=duration_ms(history_start),
                        messages=len(prior),
                    )

                # 3. assemble: persona + recalled context + history + new msg
                context_msg = ""
                if recalled:
                    context_msg = "\n\nRelevant past context:\n" + "\n---\n".join(recalled)
                msgs: list[Any] = []
                if context_msg:
                    msgs.append(SystemMessage(content=context_msg))
                msgs.extend(prior)
                msgs.append(HumanMessage(content=user_text))

                state: AgentState = {
                    "messages": msgs,
                    "cost_spent": 0.0,
                    "session_id": session_id,
                    "tool_calls_made": 0,
                    "llm_calls_made": 0,
                    "trace": trace,
                }
                result = await self._graph.ainvoke(state)
                out_msgs = result.get("messages", [])
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
                if not final:
                    final = "I wasn't able to complete this task within the iteration limit. Please try a simpler request or rephrase."

                # 4. persist to SQLite + ChromaDB
                persist_start = time.perf_counter()
                if self._chatdb:
                    self._chatdb.add_message(session_id, "user", user_text)
                    self._chatdb.add_message(session_id, "assistant", final)
                    cost_usd = result.get("cost_spent", 0.0)
                    if cost_usd > 0:
                        self._chatdb.add_cost(int(cost_usd * 1_000_000))
                if self._vector and final:
                    import uuid
                    uid = str(uuid.uuid4())
                    self._vector.add(
                        ids=[uid],
                        texts=[f"User: {user_text}\nAssistant: {final}"],
                        metas=[{"session_id": session_id}],
                    )
                trace.emit("memory_persisted", duration_ms=duration_ms(persist_start))
                trace.emit(
                    "turn_completed",
                    duration_ms=duration_ms(turn_start),
                    cost_spent=result.get("cost_spent", 0.0),
                    llm_calls=result.get("llm_calls_made", 0),
                    tool_rounds=result.get("tool_calls_made", 0),
                )
                return {
                    "text": final,
                    "cost_spent": result.get("cost_spent", 0.0),
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
