from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from typing_extensions import TypedDict

from agent.react import build_react_graph
from agent.shared import activate_session
from agent.subagents.types import SubAgentConfig, SubAgentResult
from agent.trace import TraceCollector, duration_ms
from agent.tools.computer import clear_screenshots

logger = logging.getLogger(__name__)


class SubAgentState(TypedDict, total=False):
    messages: list[Any]
    tool_calls_made: int
    llm_calls_made: int
    trace: TraceCollector
    started_at: float
    stopped_reason: str  # "max_iterations" | "max_seconds" | "cancelled" | ""
    cancel_event: Any  # asyncio.Event — cooperative cancellation token


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
        self._graphs[config.name] = build_react_graph(
            SubAgentState,
            bound_llm,
            tools,
            max_iter=config.max_iterations,
            max_secs=config.max_seconds,
            system_prompt=lambda: config.system_prompt,
            model_name=effective_model_name,
            backend=self._llm_backend,
            trace_prefix="subagent_",
            trace_context={"subagent_type": config.name},
            vision_llm=self._vision_llm,
            vision_model=self._vision_model,
            vision_failure_message="Subagent '%s' vision call failed: %s",
            vision_failure_args=(config.name,),
        )

    async def run(
        self,
        subagent_type: str,
        description: str,
        context: str = "",
        trace: TraceCollector | None = None,
        session_id: str = "",
        cancel_event: asyncio.Event | None = None,
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
            "cancel_event": cancel_event,
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
        final = extract_final_ai_text(out_msgs)
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
