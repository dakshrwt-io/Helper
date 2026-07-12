# Vision Pipeline Fix Plan

## Problem A (parent agent) — fixed

`agent/graph.py:252-263` — vision LLM response appended as separate `AIMessage` after screenshot `ToolMessage`. Main LLM sees two responses to one tool call. May ignore vision description. **FIXED** — `model_copy` overwrites ToolMessage content with vision analysis.

## Problem B (sub-agents) — active bug

`agent/subagents/manager.py:97-152` — sub-agent `agent_node` has **zero vision logic**. When `browser` / `computer_control` sub-agents call `computer_screenshot`:

1. Tool executes → returns `"Screenshot captured (1920x1080). Describe what you see."`
2. Sub-agent `agent_node` runs → **no vision check** → jumps straight to LLM call
3. Sub-agent LLM (deepseek-v4-flash, **non-vision model**) receives ToolMessage text only — no image
4. LLM **hallucinates** screen contents

**Evidence from trace** (session `98f4be38`):
- Sub-agent called `computer_screenshot` 3 times
- Zero `vision_started` / `vision_completed` events
- Sub-agent LLM responded "The screen shows a typical Windows desktop" and "LinkedIn sign-in page" — both wrong. Actual screen had LinkedIn profile open.

## Root Cause

`SubAgentManager.__init__` never receives `vision_llm` / `vision_model`. Sub-agent graphs built without vision support. No import of `inject_screenshots` or `ToolMessage` in manager.py.

## Fix — Part 2 (sub-agents)

### File 1: `agent/graph.py` — pass vision to SubAgentManager

**Location**: Lines 213-220 — `SubAgentManager(...)` constructor call

**Change**: Add two keyword arguments:
```python
vision_llm=self._vision_llm,
vision_model=self._vision_model,
```

### File 2: `agent/subagents/manager.py` — full vision support

**A. Imports** (add to existing):
```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # add ToolMessage
from agent.tools.computer import inject_screenshots                                # new
```

**B. `__init__` parameters** (lines 29-37):
```python
def __init__(
    self,
    raw_config: dict[str, Any],
    mcp_tools: list[StructuredTool],
    computer_tools: list[StructuredTool],
    llm: Any,
    llm_backend: str,
    model_name: str,
    vision_llm: Any = None,      # new
    vision_model: str = "",       # new
) -> None:
    ...
    self._vision_llm = vision_llm          # new
    self._vision_model = vision_model      # new
```

**C. `agent_node` — vision block** (after line 101, before LLM call):
```python
# ── Vision: analyse screenshot if last message is one ──
if self._vision_llm is not None and msgs and isinstance(msgs[-1], ToolMessage) and getattr(msgs[-1], "name", None) == "computer_screenshot":
    vision_start = time.perf_counter()
    if trace:
        trace.emit(
            "vision_started",
            subagent_type=subagent_type,
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
                subagent_type=subagent_type,
                duration_ms=duration_ms(vision_start),
                content=vision_text,
                usage=getattr(vision_resp, "usage_metadata", None) or {},
            )
    except Exception as exc:
        logger.warning("Subagent '%s' vision call failed: %s", subagent_type, exc)
        if trace:
            trace.emit(
                "vision_failed",
                subagent_type=subagent_type,
                duration_ms=duration_ms(vision_start),
                error_type=type(exc).__name__,
                error=str(exc),
            )
# ── end vision block ──
```

**D. `run` method** — clear screenshots before sub-agent starts (after line 233):
```python
from agent.tools.computer import clear_screenshots  # add to import block

# in run(), before graph.ainvoke:
clear_screenshots()
```

## Verification

| What | How |
|---|---|
| Sub-agent with computer tools gets vision | `browser` / `computer_control` sub-agents pass through vision check |
| Sub-agent without computer tools unaffected | Vision check `is not None` + ToolMessage gate — no screenshot calls = no-op |
| `vision_llm` is None | Guard clause skips entire block, same behavior as today |
| `clear_screenshots()` isolation | Each sub-agent `run()` starts with clean buffer |
| No duplicate ToolMessages | Same `model_copy` approach as parent — `id` preserved for add_messages dedup |
| Trace events emit correctly | `subagent_type` included in vision events for observability |

## Files Changed

| File | Change |
|---|---|
| `agent/graph.py:213-220` | Pass `vision_llm`, `vision_model` to SubAgentManager (2 lines added) |
| `agent/subagents/manager.py:1-16` | Imports: add `ToolMessage`, `inject_screenshots`, `clear_screenshots` |
| `agent/subagents/manager.py:29-34` | `__init__`: accept + store `vision_llm`, `vision_model` (2 params + 2 assignments) |
| `agent/subagents/manager.py:101` | `agent_node`: vision check block (~30 lines) between sanitization and LLM call |
| `agent/subagents/manager.py:242` | `run`: `clear_screenshots()` before graph invoke (1 line) |
