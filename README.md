# Personal AI Agent

A 24x7 personal AI assistant running on your local Windows machine, accessible via a web UI and a Telegram bot. Built with LangGraph + LangChain, powered by OpenRouter, Ollama, DeepSeek, NVIDIA NIM, or Groq (env-switchable), with filesystem tools via Model Context Protocol (MCP), desktop control (PyAutoGUI), a hierarchical sub-agent system, persistent memory (SQLite + ChromaDB), real-time execution tracing, and auto-restart via PM2.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Architecture](#architecture)
5. [Configuration](#configuration)
6. [How to Start](#how-to-start)
7. [How to Use](#how-to-use)
8. [How to Stop](#how-to-stop)
9. [How to Restart](#how-to-restart)
10. [Viewing Logs](#viewing-logs)
11. [Understanding the Code](#understanding-the-code)
12. [Troubleshooting](#troubleshooting)
13. [Extending the Agent](#extending-the-agent)

---

## Overview

The agent runs as a FastAPI web server on `http://127.0.0.1:8000` (localhost only, no auth) plus an optional Telegram bot. You chat with it through a browser UI or Telegram. It can read/write files in your workspace (via MCP), control your computer via PyAutoGUI (click, type, scroll, hotkeys), delegate complex work to specialized sub-agents, and recall past conversations via vector search.

**Trigger model:** On-message only. The agent does nothing unless you send it a message.

**Safety:** Computer-control actions (click, type, scroll, drag, hotkeys) execute immediately — no confirmation gating.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.12+ | Core runtime |
| Agent framework | LangGraph + LangChain | Graph-based agent loop |
| LLM provider | OpenRouter, Ollama, DeepSeek, NVIDIA NIM, Groq | Cloud (pay-per-token) or local (free); single env switch |
| LLM client | `langchain-openai` / `langchain-ollama` / `langchain-nvidia-ai-endpoints` | Swap backend via `LLM_BACKEND` env var |
| Tool protocol | MCP (Model Context Protocol) | Standard tool interface |
| MCP servers | `@modelcontextprotocol/server-filesystem` | File tools (Node/npx) |
| Desktop automation | PyAutoGUI + Pillow | Mouse, keyboard, screenshots |
| Sub-agents | LangGraph (nested) | Isolated specialized agents via a `task` tool |
| Memory (history) | SQLite via SQLAlchemy + aiosqlite | Per-session chat log |
| Memory (recall) | ChromaDB (persistent, local) | Vector search of past turns for context recall |
| Web server | FastAPI + Uvicorn | Async HTTP + WebSocket |
| Web UI | Vanilla HTML/JS (single page) | Chat interface, no build step |
| Telegram | python-telegram-bot v21+ | Alternate chat frontend |
| Observability | In-memory trace collector | Per-turn event stream to browser |
| Process manager | PM2 (Node.js) | Auto-restart, keep-alive, auto-start on login |
| Config | YAML + `.env` | Settings + secrets |

---

## Project Structure

```
C:\Projects\Helper\
├── .env                      # secrets + backend config — NOT committed
├── .env.example              # template for .env
├── .gitignore
├── config.yaml               # all agent, MCP, memory, web, telegram, subagent settings
├── ecosystem.config.js       # PM2 process config
├── pm2_startup.vbs           # auto-start PM2 on Windows login (copied to Startup folder)
├── run_server.py             # launcher: agent → Telegram → FastAPI
├── requirements.txt
├── package.json              # Node dep (playwright-chromium)
├── data/                     # runtime data (gitignored)
│   ├── history.db            # SQLite chat history
│   └── chroma/               # ChromaDB vector store
├── pm2_logs/                 # PM2 stdout/stderr
├── agent/                    # main package
│   ├── __init__.py
│   ├── graph.py              # LangGraph build + chat() entrypoint (the brain)
│   ├── main.py               # FastAPI app: /health, /chat WebSocket, /, lifespan
│   ├── shared.py             # Singleton: graph + chat lock + chat bus
│   ├── chat_bus.py           # Async pub-sub event bus for cross-provider events
│   ├── trace.py              # TraceCollector: per-turn event stream
│   ├── persona.md            # system prompt / personality
│   ├── mcp_manager.py        # spawn MCP servers, manage sessions + tool calls + health
│   ├── mcp_adapter.py        # convert MCP tools → LangChain StructuredTools
│   ├── chat_providers/
│   │   └── telegram.py       # Telegram bot integration
│   ├── memory/
│   │   ├── db.py             # ChatDB: SQLite history
│   │   └── vector.py         # VectorStore: ChromaDB recall + auto-recovery
│   ├── tools/
│   │   └── computer.py       # 12 PyAutoGUI tools
│   ├── subagents/
│   │   ├── __init__.py       # Package exports
│   │   ├── manager.py        # Sub-agent graph builder & runner
│   │   ├── task_tool.py      # "task" StructuredTool exposed to parent agent
│   │   └── types.py          # SubAgentConfig & SubAgentResult dataclasses
│   └── web/
│       └── index.html        # chat UI
└── tests/                    # pytest + pytest-asyncio test suite
```

---

## Architecture

### Startup flow (`run_server.py`)

```
1. load_dotenv()                       — reads .env
2. AgentGraph(config.yaml)             — instantiate
3. await graph.setup()                 — load config, spawn MCP, build LLM, build tools,
                                         build sub-agents, compile LangGraph
4. set_graph(graph)                    — store in shared singleton
5. [Optional] Telegram bot             — start polling if TELEGRAM_ENABLED=true + token set
6. create_app(graph) → FastAPI         — wire app
7. uvicorn.run(...)                    — blocks until shutdown
```

### Flow: user sends a message

```
Browser (HTML/JS) or Telegram
  → WebSocket /chat  (FastAPI, main.py)   OR   message_handler (telegram.py)
    → AgentGraph.chat() (graph.py)        [serialized by shared asyncio.Lock]
        │
        ├─ 1. VectorStore.query() → relevant past turns (ChromaDB, cosine distance < 0.6)
        ├─ 2. ChatDB.get_history() → last 20 messages (SQLite)
        ├─ 3. Assemble: [SystemMessage(recalled context)] + [history] + [HumanMessage(msg)]
        ├─ 4. Graph invoke (LangGraph):
        │       agent node (LLM call, bind_tools)
        │         ↓ route(): tool_calls? → tools node → agent → ... → END
        │       (guards: max_iterations=25, max_seconds=120)
        ├─ 5. Extract final AIMessage text (last AIMessage without tool_calls)
        ├─ 6. Persist: SQLite (user+assistant msg) + ChromaDB (vector upsert)
        └─ 7. Return {text, messages, trace}
    → Provider sends answer + streams trace events to client
  → UI renders assistant bubble + updates status bar
```

Trace events (`turn_started`, `llm_started/completed/failed`, `tool_started/completed/failed`, `memory_recall_*`, `subagent_*`, `vision_*`, `turn_completed/failed`) are emitted through the `TraceCollector` and forwarded to the browser in real time.

### LangGraph structure

```
State: {messages, session_id, tool_calls_made, llm_calls_made, trace, started_at}

      ┌──────────┐
      │  START   │
      └────┬─────┘
           ↓
      ┌──────────┐  route()
      │  agent   │ ──────────→ END   (no tool_calls, or cap hit, or max iter/time)
      └────┬─────┘
           ↓ tool_calls present
      ┌──────────┐
      │  tools   │  (ToolNode dispatches to MCP + computer StructuredTools)
      └────┬─────┘
           ↓
           └──→ agent  (loop back for next LLM turn with tool results)
```

The `agent_node` injects the persona (with current UTC datetime) as a `SystemMessage` if not present, calls `llm_with_tools.ainvoke(msgs)`, and extracts token usage. When the last message is a `computer_screenshot` ToolMessage and a Groq vision LLM is configured, the screenshot is injected as an image and interpreted by Groq before the main LLM turn.

### MCP integration

`MCPManager` (`mcp_manager.py`) spawns each MCP server as a subprocess via stdio:

- `npx -y @modelcontextprotocol/server-filesystem "${FILESYSTEM_MCP_DIR}"` → 14 file tools

Each server runs a `ClientSession` (MCP Python SDK). `MCPManager.start()` opens the session, calls `session.initialize()`, and registers tools. A background health monitor pings each server every 30s; unhealthy servers' tools are filtered out of `tool_names`.

`mcp_adapter.py` wraps each MCP tool as a `langchain_core.tools.StructuredTool` with:

- Auto-generated Pydantic args model (from the MCP tool's JSON schema)
- Async coroutine that calls `MCPManager.call_tool` and stringifies the result
- Trace emission (`tool_started` / `tool_completed` / `tool_failed`)

These StructuredTools feed into `llm.bind_tools(tools)` and `ToolNode(tools)`, so LangGraph's prebuilt ToolNode handles dispatch automatically.

### Computer control

`agent/tools/computer.py` exposes 12 PyAutoGUI tools:

- **Read-only:** `computer_screenshot`, `computer_get_screen_size`, `computer_get_mouse_position`
- **Actuating:** `computer_move_mouse`, `computer_click`, `computer_double_click`, `computer_right_click`, `computer_type_text`, `computer_press_key`, `computer_hotkey`, `computer_scroll`, `computer_drag`

All actions execute immediately — there is no confirmation step.

Screenshots are buffered (max 5, FIFO) and, when a Groq vision LLM is configured, injected as `image_url` HumanMessages so the vision model can interpret them.

### Sub-agent system

The parent agent can delegate to specialized isolated sub-agents via a single `task` tool (`agent/subagents/task_tool.py`). Each sub-agent runs its own compiled LangGraph with a subset of tools and its own iteration/time caps. The sub-agent returns only its final result, keeping the parent's context clean.

Defined sub-agents (from `config.yaml`):

| Name | Tools | Max Iter | Max Sec | Purpose |
|---|---|---|---|---|
| `filesystem` | MCP | 10 | 60 | File read/write/edit/search |
| `coding` | MCP | 20 | 180 | Code write/edit/debug/verify |
| `git` | MCP | 10 | 60 | Git operations |
| `browser` | Computer | 15 | 120 | Browser/web navigation |
| `computer_control` | Computer | 15 | 120 | Desktop GUI automation |
| `general` | MCP + Computer | 15 | 120 | Fallback: all tools |

The persona marks browser, desktop/GUI, coding, git, and multi-file tasks as **mandatory delegation** to the matching sub-agent.

### Memory

**SQLite (`ChatDB`, `db.py`):**

- `messages` table: `(id, session_id, role, content, ts)` — full chat log per session
- WAL mode, `busy_timeout=5000ms`
- `get_history(session_id, limit)` → last N messages (chronological)
- `add_turn(session_id, user_text, assistant_text)` → atomic user+assistant insert
- `export_all_turns()` → rebuild material for ChromaDB recovery

**ChromaDB (`VectorStore`, `vector.py`):**

- Persistent client at `data/chroma`, collection `chat`, cosine space
- Each completed turn (user+assistant) stored as a document with `session_id` metadata
- `query(text, top_k, session_id)` → session-filtered cosine similarity; filtered in `graph.py` by `distance < 0.6`
- Auto-recovery: on init failure, deletes + recreates the store
- `rebuild(ids, texts, metas)` → disaster recovery from SQLite turns (triggered on empty store at boot)

### Observability (trace)

`TraceCollector` (`trace.py`) is a per-turn, in-memory event log. It is scoped to the active chat request via a `contextvars.ContextVar` so nested async tool calls can emit events. Events are JSON-safe, sequenced, timestamped, and optionally pushed to an `asyncio.Queue` for streaming to the browser. Trace data is not written to logs or the database.

### Shared infrastructure (`shared.py`)

```python
_graph: AgentGraph | None = None       # Singleton agent instance
_chat_locks: dict[str, asyncio.Lock]   # Serializes turns within each session
_chat_bus: ChatBus                     # Session-scoped pub-sub
```

Web and Telegram share one `AgentGraph`. Different sessions run concurrently; events, history, traces, and vector recall remain session-scoped.

---

## Configuration

### `.env` (secrets — not committed)

```
# Backend selection: openrouter | ollama | deepseek | groq | nvidia
LLM_BACKEND=ollama

# OpenRouter (used when LLM_BACKEND=openrouter)
OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=z-ai/glm-4.5

# Ollama (used when LLM_BACKEND=ollama)
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434

# DeepSeek (used when LLM_BACKEND=deepseek)
DEEPSEEK_API_KEY=sk-REPLACE_ME
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_THINKING=false
DEEPSEEK_REASONING_EFFORT=max

# Groq - used ONLY for vision (screenshot input via Llama 4 Scout).
# Leave blank to skip vision support (screenshots use the text LLM).
GROQ_API_KEY=gsk_REPLACE_ME
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# NVIDIA AI Endpoints (used when LLM_BACKEND=nvidia)
NVIDIA_API_KEY=nvapi-REPLACE_ME
NVIDIA_MODEL=minimaxai/minimax-m3
NVIDIA_TEMPERATURE=1.0
NVIDIA_TOP_P=0.95
NVIDIA_MAX_TOKENS=8192

# Telegram chat provider
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghikl-zyx57W2v1u123ew11
TELEGRAM_ALLOWED_USERS=

WEB_PORT=8000
WEB_TOKEN=replace-with-a-long-random-token
WEB_ALLOWED_ORIGINS=
WEB_COOKIE_SECURE=false
PERSONA_PATH=agent/persona.md
DATA_DIR=data
# MCP filesystem root directory
FILESYSTEM_MCP_DIR=C:\\FILESYSTEM_MCP

# Computer control - PyAutoGUI desktop automation
COMPUTER_CONTROL_ENABLED=false
COMPUTER_CONTROL_LEASE_SECONDS=300
COMPUTER_CONTROL_RATE_LIMIT=30
```

Get your OpenRouter key at https://openrouter.ai → sign up → create key.

### LLM Backend

The agent supports five backends, selected by `LLM_BACKEND` in `.env`:

| Backend | When to use | Cost | Setup |
|---------|-------------|------|-------|
| `ollama` | Local, private, no internet | Free | Install [Ollama](https://ollama.com), `ollama pull qwen2.5:3b` |
| `openrouter` | Cloud model variety | Pay-per-token | Get API key from openrouter.ai |
| `deepseek` | DeepSeek direct API | Pay-per-token | Get API key from deepseek.com |
| `nvidia` | NVIDIA NIM endpoints | Pay-per-token | Get API key from build.nvidia.com |
| `groq` | (vision sidecar only) | Pay-per-token | Used for screenshot interpretation, not as primary backend |

**Switching backends:** Edit `LLM_BACKEND` in `.env`, then `pm2 restart ai-agent`. No code changes needed.

**Ollama setup:**

```powershell
# Install from https://ollama.com, then:
ollama pull qwen2.5:3b
# Verify:
ollama list
# Start the Ollama service (runs on port 11434):
ollama serve
```

**GPU support:** Ollama auto-detects NVIDIA GPUs. Verify with `ollama ps` — the `PROCESSOR` column should show `100% GPU`. If it shows a CPU/GPU split, the model is too large for your VRAM; use a smaller model.

**Vision (Groq):** When `GROQ_API_KEY` is set, screenshot ToolMessages are interpreted by the Groq Llama 4 Scout model. Text-only turns use the main `LLM_BACKEND`. Leave the key blank to skip vision support.

### `config.yaml` (settings)

```yaml
llm:
  backend: ${LLM_BACKEND}
  openrouter:  { model, api_key, base_url }
  ollama:      { model, base_url }
  deepseek:    { model, api_key, base_url, thinking, reasoning_effort }
  nvidia:      { model, api_key, temperature, top_p, max_completion_tokens }
  groq:        { model, api_key, base_url }

mcp:
  filesystem:
    command: npx
    args: [-y, "@modelcontextprotocol/server-filesystem", "${FILESYSTEM_MCP_DIR}"]
    transport: stdio

memory:
  sqlite: ${DATA_DIR}/history.db
  chroma:  ${DATA_DIR}/chroma

web:
  port: ${WEB_PORT}
  host: 127.0.0.1
  auth: true

telegram:
  enabled: ${TELEGRAM_ENABLED}
  token: ${TELEGRAM_BOT_TOKEN}
  allowed_users: ${TELEGRAM_ALLOWED_USERS}

computer_control:
  enabled: ${COMPUTER_CONTROL_ENABLED}

agent:
  persona_path: ${PERSONA_PATH}
  temperature: 0.3
  max_iterations: 25
  max_seconds: 120

subagents:
  enabled: true
  default_max_iterations: 10
  default_max_seconds: 60
  agents: { filesystem, coding, git, browser, computer_control, general }
```

Env vars (`${VAR}`) are expanded at load time. `MCPManager._expand` and `AgentGraph._load` both call `os.path.expandvars` recursively on the parsed YAML.

### `agent/persona.md`

Injected as the first `SystemMessage` in every conversation (loaded at `setup()` time, not per-message). Current rules:

- Think before acting; prefer facts over assumptions
- Use tools to verify; never fabricate outputs
- Mandatory delegation of browser/desktop/coding/git/multi-file tasks to sub-agents via the `task` tool
- Concise, mobile-friendly communication

---

## How to Start

### First-time setup

```powershell
cd "C:\Projects\Helper"
pip install -r requirements.txt
npm install        # for MCP filesystem server (npx)
Copy-Item .env.example .env
# Edit .env: set LLM_BACKEND + API key + FILESYSTEM_MCP_DIR
```

### Starting the agent (PM2 — recommended)

```powershell
cd "C:\Projects\Helper"
pm2 start ecosystem.config.js
```

This launches `run_server.py`, auto-restart on crash, logs to `pm2_logs/`. Takes ~15s to boot (MCP server spawn + tool listing + sub-agent graph compilation).

Verify:

```powershell
pm2 ls                       # status should be "online"
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
# → {"status":"ok","tools":27}
```

Open browser: `http://127.0.0.1:8000`

### Starting without PM2 (manual)

```powershell
cd "C:\Projects\Helper"
python run_server.py
```

Stops when you close the terminal. Use only for debugging.

### Auto-start on Windows login

`pm2_startup.vbs` was copied to `C:\Users\<you>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\`. On every login it runs `pm2 resurrect` (silently restores the saved process list). `pm2 save` must have been run at least once for this to work.

To re-trigger after changing config:

```powershell
pm2 delete ai-agent
pm2 start ecosystem.config.js
pm2 save                    # update the saved snapshot
```

---

## How to Use

### Web UI

1. Open `http://127.0.0.1:8000` in any browser.
2. Type a message, hit Enter or click Send.
3. You'll see "agent thinking..." while it runs, plus live trace events.
4. The assistant's reply appears in a bubble.
5. Status bar shows connection state and available controls.

### Telegram

1. Start a chat with your bot.
2. Send `/start` to verify it's alive (shows tool count + model).
3. Send any message. The bot replies, splitting long answers into multiple messages.
4. Commands: `/start`, `/help`, `/reset`, `/desktop_on`, `/desktop_off`.
5. Access control: populate `TELEGRAM_ALLOWED_USERS` with comma-separated Telegram user IDs. Empty denies all access.

### Example prompts

- **"List the files in the current directory"** → delegates to `filesystem` sub-agent
- **"Read the contents of README.md"** → `filesystem` sub-agent
- **"Open Brave browser and go to youtube.com"** → `browser` sub-agent
- **"Open Notepad and type a message"** → `computer_control` sub-agent
- **"Clone this repo and fix the login bug"** → `git` then `coding` sub-agents
- **"What did we talk about earlier?"** → recall pulls past turns from ChromaDB

### Tool categories available

- **Filesystem (14 MCP tools):** `read_file`, `read_text_file`, `read_media_file`, `read_multiple_files`, `write_file`, `edit_file`, `create_directory`, `list_directory`, `list_directory_with_sizes`, `directory_tree`, `move_file`, `search_files`, `get_file_info`, `list_allowed_directories`
- **Computer (12 tools):** 3 read-only + 9 destructive (see [Computer control](#computer-control))
- **Sub-agent (1 tool):** `task` — delegates to a specialized sub-agent

Total depends on config: 14 (MCP only) / 26 (MCP + computer) / 27 (MCP + computer + task).

### Multi-turn context

Last 20 SQLite messages + up to 3 vector-recalled turns from the same session are sent as context each turn.

---

## How to Stop

### Temporary stop (process stays registered in PM2)

```powershell
pm2 stop ai-agent
```

Health endpoint goes offline. Use `pm2 start ai-agent` to resume.

### Full stop and remove from PM2

```powershell
pm2 delete ai-agent
```

`pm2 save` to update the snapshot (otherwise `pm2 resurrect` on next login will try to restore a dead process — harmless, it just fails).

### Stop everything (PM2 daemon itself)

```powershell
pm2 kill
```

Kills the PM2 daemon and all managed processes. PM2 stays installed; just not running.

### Disable auto-start on login

```powershell
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\pm2_startup.vbs"
```

### Stop the manual (non-PM2) run

Press `Ctrl+C` in the terminal running `run_server.py`.

---

## How to Restart

### After changing config.yaml or .env

```powershell
cd "C:\Projects\Helper"
pm2 restart ai-agent
```

PM2 kills the process and spawns a fresh one. Takes ~15s to boot MCP servers + sub-agent graphs.

### After changing code (Python files)

Same — `pm2 restart ai-agent`. PM2 does NOT watch files by default (`watch: false` in `ecosystem.config.js`) to avoid restart loops during edits.

### Hard reset (clear memory)

To wipe all chat history and vector memory:

```powershell
pm2 stop ai-agent
Remove-Item data\history.db -Force -ErrorAction SilentlyContinue
Remove-Item data\chroma -Recurse -Force -ErrorAction SilentlyContinue
pm2 start ai-agent
```

Fresh SQLite + ChromaDB will be created on next boot. If SQLite still has turns when ChromaDB is empty, the vector store is rebuilt automatically at startup.

---

## Viewing Logs

### PM2 logs (real-time)

```powershell
pm2 logs ai-agent             # live, all logs
pm2 logs ai-agent --lines 50  # last 50 lines
pm2 logs ai-agent --err       # errors only
```

### Log files on disk

```
pm2_logs\agent.out.log    # stdout (uvicorn access logs, MCP server output)
pm2_logs\agent.err.log    # stderr (Python logging, tracebacks, MCP stderr)
```

### Key log lines to look for

- `MCP server 'filesystem' started, 14 tools: [...]` — MCP boot OK
- `AgentGraph ready — model=... backend=... tools=...` — graph compiled
- `Telegram bot polling started` — Telegram online
- `Starting web server at http://127.0.0.1:8000` — uvicorn ready
- `Application startup complete.` — FastAPI lifespan done, /health is live
- `WS connected` / `WS disconnected` — chat session lifecycle
- `chat error` — exception during a turn (look at the following traceback)
- `MCP server '...' health check failed` — server unhealthy, tools filtered

---

## Understanding the Code

### `agent/graph.py` — AgentGraph

The brain. Key parts:

**`AgentState` (TypedDict):** graph state schema. Fields: `messages`, `session_id`, `tool_calls_made`, `llm_calls_made`, `trace`, `started_at`.

**`setup()` (async):** the only entrypoint to initialize everything.

1. `_load()` — parse `config.yaml`, expand env vars, load `persona.md`, init `ChatDB` + `VectorStore`. If vector store is empty but SQLite has turns, rebuilds ChromaDB from SQLite.
2. `mcp.start()` — spawn MCP servers + start health monitor.
3. `_make_llm()` — build LLM client (5 backends, single env switch). Also builds the optional Groq vision LLM.
4. `build_langchain_tools(mcp)` — wrap MCP tools as LangChain tools.
5. Optionally add computer-control tools (12 PyAutoGUI tools).
6. Optionally build sub-agent graphs + register the `task` tool.
7. `llm.bind_tools(tools)` + `ToolNode(tools)`.
8. `_build_graph()` — compile LangGraph.

**`_build_graph()`:** defines:

- `agent_node` — injects persona (+ current UTC time) as SystemMessage if absent, optionally runs Groq vision on screenshot ToolMessages, calls `llm_with_tools.ainvoke(msgs)`, extracts token usage, surfaces reasoning content.
- `tools_node` — delegates to ToolNode (which calls MCP + computer tools via StructuredTool adapters), increments `tool_calls_made`.
- `route()` — conditional edge from agent. Returns `"tools"` if the last AIMessage has `tool_calls`, else `END`. Also returns `END` when `max_iterations` (25) or `max_seconds` (120) is reached.

Edges: `START → agent`, `agent →(route)→ tools | END`, `tools → agent` (loop back). Compiled with `g.compile()`.

**`chat(user_text, session_id, trace_queue)` (async):** the user-facing API. One call = one turn. Steps:

1. **Vector recall** — `VectorStore.query(user_text, top_k=3)`, filter by cosine distance < 0.6.
2. **History load** — `ChatDB.get_history(session_id, limit=20)`.
3. **Assemble** — `[SystemMessage(recalled context if any)] + prior history + [HumanMessage(user_text)]`.
4. **Graph invoke** — `self._graph.ainvoke(state)`. LangGraph runs agent→tools→agent→... until route returns END.
5. **Extract final** — scan `out_msgs` in reverse for the last AIMessage without `tool_calls`.
6. **Persist** — save user+assistant msg to SQLite (`add_turn`), upsert the turn to ChromaDB.
7. **Return** — `{text, messages, trace}`.

**`close()`** — calls `mcp.stop()` to tear down MCP subprocesses. Called by `run_server.py` on shutdown.

### `agent/mcp_manager.py` — MCPManager

Spawns and talks to MCP servers. Key methods:

- `start()` — reads `config.yaml` `mcp:` section, for each server builds `StdioServerParameters`, calls `stdio_client(params)`, opens a `ClientSession`, calls `session.initialize()`, registers tools. Starts the health monitor.
- `list_tools_async()` — re-fetches tools from all live sessions.
- `call_tool(name, args)` — looks up owning server, checks health, calls `session.call_tool(name, arguments=args)`. Returns MCP `CallToolResult`.
- `_health_loop()` — every 30s, pings each session via `session.list_tools()` with a 10s timeout; flips `_healthy` flags. `tool_names` and `tool_definitions` filter out unhealthy servers' tools.
- `stop()` — stops health monitor, closes the `AsyncExitStack` (subprocesses killed).

### `agent/mcp_adapter.py` — MCP → LangChain bridge

- `_schema_to_pydantic(name, schema)` — reads MCP tool's `inputSchema` (JSON Schema), builds a Pydantic `BaseModel` with the right field types and required/optional flags.
- `_make_tool(mcp_tool, mgr)` — creates a `StructuredTool.from_function` with an async coroutine that calls `mgr.call_tool(name, kwargs)`, joins text content, and emits trace events.
- `build_langchain_tools(mgr)` — fetches all MCP tools, wraps each. Returns a `list[StructuredTool]`.

### `agent/tools/computer.py` — Computer control

12 PyAutoGUI StructuredTools. See [Computer control](#computer-control). Key internals:

- `_push_screenshot` / `_pop_screenshot` / `clear_screenshots` / `inject_screenshots` — FIFO screenshot buffer (max 5) for multimodal injection.
- `_wrap_tool(name, desc, args_schema, execute)` — builds a trace-aware StructuredTool around an async executor.

### `agent/subagents/` — Sub-agent system

- `types.py` — `SubAgentConfig` (name, description, system_prompt, tools, model, max_iterations, max_seconds) and `SubAgentResult` dataclasses.
- `manager.py` — `SubAgentManager` parses `config.yaml` `subagents.agents:` into configs, builds a compiled LangGraph per sub-agent (same agent/tools/route pattern as the parent, with the sub-agent's own caps), and runs them via `run(subagent_type, description, context, trace)`.
- `task_tool.py` — `build_task_tool(manager, chatdb)` returns the `task` StructuredTool the parent agent calls to delegate. It emits `tool_started`/`tool_completed` trace events and returns a JSON summary of the sub-agent result.

### `agent/main.py` — FastAPI server

- `lifespan` context manager — logs graph readiness on startup, shutdown message on exit.
- `GET /` — serves `agent/web/index.html`.
- `POST /auth/login` and `/auth/logout` manage signed HttpOnly web sessions using `WEB_TOKEN`.
- `GET /health` — minimal liveness status.
- `WebSocket /chat` — requires authenticated session and valid Origin. Session IDs are server-owned.

### `agent/chat_providers/telegram.py` — Telegram bot

- `build_bot(token)` — registers `/start`, `/help`, `/reset`, `/desktop_on`, `/desktop_off` + text handler.
- `message_handler` — checks access whitelist, derives `session_id = "telegram_{chat_id}"`, publishes session-scoped events, runs `graph.chat` under the per-session lock with a 180s timeout, and sends the reply (split into ≤4000-byte chunks).
- Access control: `TELEGRAM_ALLOWED_USERS` comma-separated user IDs; empty denies all access.
- Retry logic for `RetryAfter` (rate limit), `TimedOut`, `NetworkError`.

### `agent/memory/db.py` — ChatDB

SQLAlchemy ORM over SQLite. Two tables:

- `messages`: `(id PK, session_id idx, role, content, ts default UTC now)`. `add_message` inserts one row; `add_turn` inserts user+assistant atomically; `get_history` returns last N by id desc, then reverses to chronological.

WAL mode + `busy_timeout=5000` set on every connection. `export_all_turns()` reconstructs user/assistant pairs for ChromaDB rebuilds.

### `agent/memory/vector.py` — VectorStore

Thin wrapper over `chromadb.PersistentClient`. Single collection `chat`, cosine space.

- `add(ids, texts, metas)` — upsert. Each completed turn stored as `"User: ...\nAssistant: ..."` with `session_id` metadata.
- `query(text, top_k, session_id)` — applies Chroma session metadata filter. Results are filtered in `graph.py` by `distance < 0.6`.
- `count()` — empty-store guard before querying.
- `_recover_store()` — on init failure, deletes + recreates the directory.
- `rebuild(ids, texts, metas, batch_size=500)` — deletes collection, recreates, batch-adds. Used for disaster recovery from SQLite.

### `agent/trace.py` — TraceCollector

Per-turn, in-memory event log scoped via `contextvars.ContextVar`. Events are JSON-safe, sequenced, timestamped. `emit()` appends to `events` and optionally pushes to an `asyncio.Queue` for streaming. `activate_trace(trace)` context manager makes a trace available to nested async tool calls. `duration_ms(start)` and `display_content(content)` are helpers.

### `agent/chat_bus.py` — ChatBus

Async session-scoped pub-sub. `subscribe(session_id, callback)` receives only matching `publish(session_id, event)` calls.

### `agent/shared.py` — Singletons

`_graph` (AgentGraph), per-session chat locks, `_chat_bus` (ChatBus), and active-session context. `sanitize_aimessage(m)` strips reasoning metadata from history.

### `ecosystem.config.js` — PM2 config

- `script: run_server.py` — launched with the system Python (edit to point at your venv if needed).
- `autorestart: true`, `max_restarts: 10`, `restart_delay: 3000` — crash recovery with backoff.
- `max_memory_restart: 500M` — memory leak guard.
- `watch: false` — no file watching. Manual restart only.
- Logs to `pm2_logs/agent.{out,err}.log`, timestamped.

### `pm2_startup.vbs`

Windows can't easily run `pm2 startup` (needs admin). This VBS script runs `cmd /c pm2 resurrect` silently (window hidden). Copied to the user's `Startup` folder, so Windows runs it on every login.

---

## Troubleshooting

### `GET /health` returns "Unable to connect"

- `pm2 ls` — is `ai-agent` `online`? If `errored`, check `pm2 logs ai-agent --err`.
- If just `stopped`, run `pm2 start ai-agent`.
- If boot is slow (MCP servers take ~15s to spawn on first run), wait and retry.

### MCP server fails to start

- **filesystem:** check `npx -y @modelcontextprotocol/server-filesystem --help` works. Node must be installed.
- **"Connection closed"** in logs — the MCP server subprocess crashed. Check that the npm package name in `config.yaml` is exactly right and `FILESYSTEM_MCP_DIR` exists.
- **"health check failed"** — server became unresponsive; its tools are filtered out until it recovers.

### Chat returns empty text

- The agent made only tool calls, never a final AIMessage without `tool_calls`. Shouldn't happen (route sends to END after max_iterations). Check `pm2 logs` for the last turn.

### "OPENROUTER_API_KEY" error on boot

- Only applies when `LLM_BACKEND=openrouter`. Check `.env` has `LLM_BACKEND=openrouter` and a valid `OPENROUTER_API_KEY`. If using another backend, ensure `LLM_BACKEND` matches.

### Ollama errors / connection refused

- Verify Ollama is running: `ollama serve` (or check `http://127.0.0.1:11434/api/tags` in browser).
- Verify model is pulled: `ollama list` should show your configured model.
- Check `OLLAMA_BASE_URL` in `.env` matches `http://127.0.0.1:11434`.
- If Ollama is slow (CPU-only), increase timeout patience.

### Telegram bot not responding

- Verify `TELEGRAM_ENABLED=true` and `TELEGRAM_BOT_TOKEN` is set in `.env`.
- Check `pm2 logs ai-agent` for `Telegram bot polling started`.
- If `TELEGRAM_ALLOWED_USERS` is set, ensure your Telegram user ID is in the list. Send `/start` — if access is denied, the bot replies with your user ID.
- The bot uses a 180s timeout per turn; complex requests may take a while.

### ChromaDB error on Windows

- Needs Microsoft VC++ runtime. Install "Visual C++ Redistributable for Visual Studio 2015-2022" from Microsoft. Reboot. Retry.
- On corruption, the store auto-recovers (deletes + recreates) at startup; if SQLite has turns, ChromaDB is rebuilt from them.

### PM2 says "App not found" on resurrect

- You need `pm2 save` after `pm2 start`. Run `pm2 start ecosystem.config.js && pm2 save`.

---

## Extending the Agent

### Add another MCP server

Edit `config.yaml`:

```yaml
mcp:
  filesystem: { ... }
  github:
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-github"
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_TOKEN}
    transport: stdio
```

Add `GITHUB_TOKEN=...` to `.env`. `MCPManager._parse_servers` picks it up; `build_langchain_tools` wraps its tools; the LLM gets them via `bind_tools`. Restart: `pm2 restart ai-agent`.

### Add a custom (non-MCP) tool

Create `agent/tools/my_tool.py`:

```python
from langchain_core.tools import tool

@tool
def get_current_time() -> str:
    """Returns the current UTC time as ISO 8601."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
```

In `graph.py:setup()`, after `tools = await build_langchain_tools(self.mcp)`, add:

```python
from agent.tools.my_tool import get_current_time
tools.append(get_current_time)
```

The ToolNode dispatches it like any MCP tool. Restart.

### Add a new sub-agent type

Add an entry under `subagents.agents:` in `config.yaml` with `description`, `system_prompt`, `tools` (`[mcp]`, `[computer]`, or `[mcp, computer]`), `max_iterations`, `max_seconds`. `SubAgentManager._parse_configs` picks it up and builds its graph at next boot. The `task` tool's description auto-lists it.

### Change the persona

Edit `agent/persona.md`. `pm2 restart ai-agent`. Takes effect on next `setup()` (not per-message).

### Change the model

- **Ollama:** `ollama pull <model>`, then edit `.env`: `OLLAMA_MODEL=<model>`, `pm2 restart ai-agent`.
- **OpenRouter:** Edit `.env`: `MODEL=<openrouter-model-id>`. Restart.
- **DeepSeek / NVIDIA:** Edit the corresponding `*_MODEL` in `.env`. Restart.

Ensure the model supports tool-calling.

### Switch between backends

Edit `.env`: set `LLM_BACKEND=openrouter|ollama|deepseek|nvidia`. `pm2 restart ai-agent`. All config (model name, API key, base URL) is read from the corresponding section in `.env`/`config.yaml`.

### Web authentication

Set a long random `WEB_TOKEN`. Browser login creates a signed HttpOnly session cookie. Set `WEB_COOKIE_SECURE=true` behind HTTPS and list reverse-proxy origins in `WEB_ALLOWED_ORIGINS`.

### Move to a VPS for true 24x7

Local Windows machine sleeps → agent dies. For genuine 24x7:

1. Deploy to a Linux VPS (DigitalOcean / Hetzner / etc.).
2. Install Node, Python, clone this repo, `pip install -r requirements.txt`, `npm install`.
3. `pm2 start ecosystem.config.js && pm2 save`.
4. On Linux, `pm2 startup` works natively (creates a systemd service) — no VBS hack needed.
5. Point a domain at the VPS, add HTTPS via Caddy/nginx, optionally add auth.
6. Update `config.yaml` MCP filesystem path to the VPS workspace. Note: PyAutoGUI desktop control is Windows-only.

---

## Quick Reference — All Commands

```powershell
# Start / stop
pm2 start ecosystem.config.js          # launch
pm2 stop ai-agent                       # pause (kept registered)
pm2 restart ai-agent                    # restart after config/code change
pm2 delete ai-agent                     # remove from PM2
pm2 kill                                # kill PM2 daemon entirely

# Status
pm2 ls                                  # process table
pm2 logs ai-agent                       # live logs
pm2 logs ai-agent --err --lines 30      # last 30 error lines
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing   # health JSON

# Memory wipe
pm2 stop ai-agent
Remove-Item data\history.db -Force
Remove-Item data\chroma -Recurse -Force
pm2 start ai-agent

# Manual run (no PM2)
python run_server.py

# Auto-start (re-copy script after Windows profile reset)
Copy-Item pm2_startup.vbs "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\"

# Disable auto-start
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\pm2_startup.vbs"

# Tests
python -m pytest tests/ -v              # unit tests (no API keys needed)
python -m pytest tests/ -q              # quiet
```
