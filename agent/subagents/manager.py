from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.shared import activate_session, sanitize_aimessage
from agent.subagents.types import SubAgentConfig, SubAgentResult
from agent.trace import TraceCollector, activate_trace, display_content, duration_ms
from agent.tools.computer import clear_screenshots, inject_screenshots, strip_image_content

logger = logging.getLogger(__name__)


class SubAgentState(TypedDict, total=False):
    messages: list[Any]
    tool_calls_made: int
    llm_calls_made: int
    trace: TraceCollector
    started_at: float
    stopped_reason: str  # "max_iterations" | "max_seconds" | ""


class SubAgentManager:
    def __init__(
        self,
        raw_config: dict[str, Any],
        mcp_tools: list[StructuredTool],
        computer_tools: list[StructuredTool],
        llm: Any,
        llm_backend: str,
        model_name: str,
        vision_llm: Any = None,
        vision_model: str = "",
    ) -> None:
        self._mcp_tools = mcp_tools
        self._computer_tools = computer_tools
        self._llm = llm
        self._llm_backend = llm_backend
        self._model_name = model_name
        self._vision_llm = vision_llm
        self._vision_model = vision_model

        self._llm_cfg = raw_config.get("llm", {})
        self._temperature = float(raw_config.get("agent", {}).get("temperature", 0.3))

        self._configs: dict[str, SubAgentConfig] = {}
        self._graphs: dict[str, Any] = {}
        self._bound_llms: dict[str, Any] = {}
        self._tool_nodes: dict[str, ToolNode] = {}
        self._custom_llms: dict[str, Any] = {}  # cache: model_name → LLM instance

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
        from agent.llm_factory import build_llm

        tools = self._get_tools_for(config)

        # Per-subagent model override: build dedicated LLM if config.model is set
        if config.model and config.model != self._model_name:
            if config.model not in self._custom_llms:
                custom_llm, _ = build_llm(
                    llm_cfg=self._llm_cfg,
                    backend=self._llm_backend,
                    model_name=config.model,
                    temperature=self._temperature,
                )
                self._custom_llms[config.model] = custom_llm
            effective_llm = self._custom_llms[config.model]
            effective_model_name = config.model
        else:
            effective_llm = self._llm
            effective_model_name = self._model_name

        bound_llm = effective_llm.bind_tools(tools) if tools else effective_llm
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

            # ── Vision: inject screenshot images for vision model; strip before acting model ──
            if self._vision_llm is not None and msgs and isinstance(msgs[-1], ToolMessage) and getattr(msgs[-1], "name", None) == "computer_screenshot":
                # Inject screenshot image into message list so vision model sees pixels
                msgs = inject_screenshots(msgs)
                vision_start = time.perf_counter()
                if trace:
                    trace.emit(
                        "vision_started",
                        subagent_type=subagent_type,
                        model=self._vision_model,
                        backend="vision",
                    )
                try:
                    vision_resp = await self._vision_llm.ainvoke(msgs)
                    vision_text = display_content(vision_resp.content)
                    # Append vision description as additional context
                    msgs.append(
                        HumanMessage(content=f"[Screen analysis]\n{vision_text}")
                    )
                    # Strip image blocks so text-only acting model does not reject them
                    msgs = strip_image_content(msgs)
                    if trace:
                        trace.emit(
                            "vision_completed",
                            subagent_type=subagent_type,
                            duration_ms=duration_ms(vision_start),
                            content=vision_text,
                            usage=getattr(vision_resp, "usage_metadata", None) or {},
                        )
                except Exception as exc:
                    logger.warning("Subagent '%s' vision call failed: %s", subagent_type, exc)
                    # Strip images even on failure so acting model can proceed
                    msgs = strip_image_content(msgs)
                    if trace:
                        trace.emit(
                            "vision_failed",
                            subagent_type=subagent_type,
                            duration_ms=duration_ms(vision_start),
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
            # ── end vision block ──

            llm_call = state.get("llm_calls_made", 0) + 1

            start = time.perf_counter()
            if trace:
                trace.emit(
                    "subagent_llm_started",
                    subagent_type=subagent_type,
                    llm_call=llm_call,
                    model=effective_model_name,
                    backend=self._llm_backend,
                    message_count=len(msgs),
                )
            try:
                resp = await bound_llm.ainvoke(msgs)
                cancel = state.get("cancel_event")
                if cancel and cancel.is_set():
                    return {
                        "messages": state.get("messages", []),
                        "stopped_reason": "cancelled",
                    }
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

            # Detect forced stop: LLM wants more tool calls but limits prevent next iteration
            tc = state.get("tool_calls_made", 0)
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
        session_id: str = "",
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
        clear_screenshots()
        if trace:
            trace.emit(
                "subagent_started",
                subagent_type=subagent_type,
                description=description[:200],
            )

        try:
            with activate_trace(trace):
                if session_id:
                    with activate_session(session_id):
                        result = await graph.ainvoke(state)
                else:
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
        elif final.startswith("[Thinking]"):
            logger.warning(
                "Reasoning content leaked into subagent final answer — "
                "this indicates a bug in reasoning-content handling"
            )

        iterations = result.get("tool_calls_made", 0)
        llm_calls = result.get("llm_calls_made", 0)
        stopped_reason = result.get("stopped_reason", "")
        completed = not bool(stopped_reason)
        success = True

        if trace:
            trace.emit(
                "subagent_completed",
                subagent_type=subagent_type,
                duration_ms=duration_ms(start),
                iterations=iterations,
                llm_calls=llm_calls,
                output_preview=final[:200],
                completed=completed,
                stopped_reason=stopped_reason or None,
            )

        return SubAgentResult(
            subagent_type=subagent_type,
            output=final,
            iterations=iterations,
            llm_calls=llm_calls,
            success=success,
            completed=completed,
            stopped_reason=stopped_reason or None,
        )
