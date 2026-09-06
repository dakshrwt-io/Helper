from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.subagents.manager import SubAgentManager
from agent.subagents.types import SubAgentResult
from agent.shared import get_active_session_id, get_cancel_event
from agent.trace import duration_ms, get_active_trace


class TaskArgs(BaseModel):
    subagent_type: str = Field(
        ...,
        description="Which specialized subagent to delegate to. Available values depend on configuration.",
    )
    description: str = Field(
        ...,
        description="Detailed description of the task. Be specific about what you need done and what a successful result looks like.",
    )
    context: str = Field(
        default="",
        description="Optional additional context, file contents, or reference information the subagent needs.",
    )


def build_task_tool(manager: SubAgentManager) -> StructuredTool:

    async def _run(subagent_type: str, description: str, context: str = "") -> str:
        trace = get_active_trace()
        session_id = get_active_session_id() or ""
        start = time.perf_counter()
        if trace:
            trace.emit(
                "tool_started",
                tool_name="task",
                server="subagent",
                arguments={
                    "subagent_type": subagent_type,
                    "description": description[:200],
                },
            )

        try:
            result: SubAgentResult = await manager.run(
                subagent_type=subagent_type,
                description=description,
                context=context,
                trace=trace,
                session_id=session_id,
                cancel_event=get_cancel_event(session_id) if session_id else None,
            )
        except Exception as exc:
            if trace:
                trace.emit(
                    "tool_failed",
                    tool_name="task",
                    server="subagent",
                    duration_ms=duration_ms(start),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise

        output: dict[str, Any] = {
            "subagent_type": result.subagent_type,
            "success": result.success,
            "completed": result.completed,
            "iterations": result.iterations,
            "llm_calls": result.llm_calls,
            "output": result.output,
        }
        if result.error:
            output["error"] = result.error
        if result.stopped_reason:
            output["stopped_reason"] = result.stopped_reason

        if trace:
            trace.emit(
                "tool_completed",
                tool_name="task",
                server="subagent",
                duration_ms=duration_ms(start),
                output=json.dumps(output, ensure_ascii=False),
            )

        return json.dumps(output, ensure_ascii=False)

    agent_list = ", ".join(manager.agent_names) if manager.agent_names else "general"

    return StructuredTool.from_function(
        coroutine=_run,
        name="task",
        description=(
            "Delegate a complex, multi-step task to a specialized subagent that works in isolation "
            "and returns only the final result. Use this to keep your context clean when a task "
            "would require many tool calls.\n\n"
            "Available subagent types: " + agent_list + "\n\n"
            "Choose the right subagent for the job. Provide clear instructions in 'description'. "
            "Use 'context' to pass relevant background info like file contents or error messages.\n\n"
            "The subagent runs independently and returns one final answer. You can then continue "
            "with the result."
        ),
        args_schema=TaskArgs,
    )
