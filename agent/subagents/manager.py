from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.shared import sanitize_aimessage
from agent.subagents.types import SubAgentConfig, SubAgentResult
from agent.trace import TraceCollector, activate_trace, display_content, duration_ms

logger = logging.getLogger(__name__)


class SubAgentState(TypedDict, total=False):
    messages: list[Any]
    tool_calls_made: int
    llm_calls_made: int
    trace: TraceCollector
    started_at: float


class SubAgentManager:
    def __init__(
        self,
        raw_config: dict[str, Any],
        mcp_tools: list[StructuredTool],
        computer_tools: list[StructuredTool],
        llm: Any,
        llm_backend: str,
        model_name: str,
    ) -> None:
        self._mcp_tools = mcp_tools
        self._computer_tools = computer_tools
        self._llm = llm
        self._llm_backend = llm_backend
        self._model_name = model_name

        self._configs: dict[str, SubAgentConfig] = {}
        self._graphs: dict[str, Any] = {}
        self._bound_llms: dict[str, Any] = {}
        self._tool_nodes: dict[str, ToolNode] = {}

        self._parse_configs(raw_config)
        self._build_all_graphs()

    @property
    def configs(self) -> dict[str, SubAgentConfig]:
        return dict(self._configs)

    @property
    def agent_names(self) -> list[str]:
        return list(self._configs.keys())

    def _parse_configs(self, raw: dict[str, Any]) -> None:
        sub_cfg = raw.get("subagents", {})
        if not sub_cfg.get("enabled", True):
            return

        defaults: dict[str, Any] = {
            "max_iterations": int(sub_cfg.get("default_max_iterations", 10)),
            "max_seconds": float(sub_cfg.get("default_max_seconds", 60.0)),
            "tools": [],
            "model": None,
        }
        for name, data in sub_cfg.get("agents", {}).items():
            self._configs[name] = SubAgentConfig.from_dict(name, data, defaults)

    def _get_tools_for(self, config: SubAgentConfig) -> list[StructuredTool]:
        tools: list[StructuredTool] = []
        requested = set(config.tools)
        if "mcp" in requested:
            tools.extend(self._mcp_tools)
        if "computer" in requested:
            tools.extend(self._computer_tools)
        return tools

    def _build_all_graphs(self) -> None:
        for name, config in self._configs.items():
            self._build_graph_for(config)

    def _build_graph_for(self, config: SubAgentConfig) -> None:
        tools = self._get_tools_for(config)
        bound_llm = self._llm.bind_tools(tools) if tools else self._llm
        self._bound_llms[config.name] = bound_llm
        self._tool_nodes[config.name] = ToolNode(tools) if tools else None

        max_iter = config.max_iterations
        max_secs = config.max_seconds
        subagent_type = config.name

        async def agent_node(state: SubAgentState) -> dict[str, Any]:
            msgs = list(state.get("messages", []))
            if not msgs or not isinstance(msgs[0], SystemMessage):
                msgs = [SystemMessage(content=config.system_prompt)] + msgs
            msgs = [sanitize_aimessage(m) if isinstance(m, AIMessage) else m for m in msgs]

            trace = state.get("trace")
            llm_call = state.get("llm_calls_made", 0) + 1

            start = time.perf_counter()
            if trace:
                trace.emit(
                    "subagent_llm_started",
                    subagent_type=subagent_type,
                    llm_call=llm_call,
                    model=self._model_name,
                    backend=self._llm_backend,
                    message_count=len(msgs),
                )
            try:
                resp = await bound_llm.ainvoke(msgs)
            except Exception as exc:
                if trace:
                    trace.emit(
                        "subagent_llm_failed",
                        subagent_type=subagent_type,
                        llm_call=llm_call,
                        duration_ms=duration_ms(start),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                raise

            reasoning = getattr(resp, "additional_kwargs", {}) or {}
            reasoning_text = reasoning.get("reasoning_content", "")
            if reasoning_text and trace:
                trace.emit("subagent_reasoning", subagent_type=subagent_type, content=reasoning_text)

            usage = getattr(resp, "usage_metadata", None)

            if trace:
                trace.emit(
                    "subagent_llm_completed",
                    subagent_type=subagent_type,
                    llm_call=llm_call,
                    duration_ms=duration_ms(start),
                    content=display_content(resp.content),
                    tool_calls=getattr(resp, "tool_calls", None) or [],
                    usage=usage or {},
                )

            return {
                "messages": msgs + [resp],
                "tool_calls_made": state.get("tool_calls_made", 0),
                "llm_calls_made": llm_call,
            }

        async def tools_node(state: SubAgentState) -> dict[str, Any]:
            tool_node = self._tool_nodes[config.name]
            if tool_node is None:
                return {}
            with activate_trace(state.get("trace")):
                out = await tool_node.ainvoke(state)
            new_msgs = out.get("messages", [])
            msgs = list(state.get("messages", [])) + new_msgs
            return {
                "messages": msgs,
                "tool_calls_made": state.get("tool_calls_made", 0) + 1,
            }

        def route(state: SubAgentState) -> str:
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

        g = StateGraph(SubAgentState)
        g.add_node("agent", agent_node)
        g.add_node("tools", tools_node)
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        self._graphs[config.name] = g.compile()

    async def run(
        self,
        subagent_type: str,
        description: str,
        context: str = "",
        trace: TraceCollector | None = None,
    ) -> SubAgentResult:
        config = self._configs.get(subagent_type)
        if config is None:
            available = ", ".join(self._configs.keys()) or "(none)"
            return SubAgentResult(
                subagent_type=subagent_type,
                output=f"No subagent named '{subagent_type}'. Available: {available}",
                iterations=0,
                llm_calls=0,
                success=False,
                error=f"Unknown subagent: {subagent_type}",
            )

        graph = self._graphs.get(subagent_type)
        if graph is None:
            return SubAgentResult(
                subagent_type=subagent_type,
                output=f"Subagent '{subagent_type}' is not compiled.",
                iterations=0,
                llm_calls=0,
                success=False,
                error="Graph not compiled",
            )

        messages: list[Any] = []
        if context:
            messages.append(HumanMessage(content=f"Additional context:\n{context}"))
        messages.append(HumanMessage(content=description))

        state: SubAgentState = {
            "messages": messages,
            "tool_calls_made": 0,
            "llm_calls_made": 0,
            "trace": trace,
            "started_at": time.perf_counter(),
        }

        start = time.perf_counter()
        if trace:
            trace.emit(
                "subagent_started",
                subagent_type=subagent_type,
                description=description[:200],
            )

        try:
            with activate_trace(trace):
                result = await graph.ainvoke(state)
        except Exception as exc:
            logger.exception("Subagent '%s' crashed", subagent_type)
            if trace:
                trace.emit(
                    "subagent_failed",
                    subagent_type=subagent_type,
                    duration_ms=duration_ms(start),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            return SubAgentResult(
                subagent_type=subagent_type,
                output=f"Subagent '{subagent_type}' crashed: {exc}",
                iterations=0,
                llm_calls=0,
                success=False,
                error=str(exc),
            )

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
            final = "Subagent completed but produced no text output."

        iterations = result.get("tool_calls_made", 0)
        llm_calls = result.get("llm_calls_made", 0)
        success = True

        if trace:
            trace.emit(
                "subagent_completed",
                subagent_type=subagent_type,
                duration_ms=duration_ms(start),
                iterations=iterations,
                llm_calls=llm_calls,
                output_preview=final[:200],
            )

        return SubAgentResult(
            subagent_type=subagent_type,
            output=final,
            iterations=iterations,
            llm_calls=llm_calls,
            success=success,
        )
