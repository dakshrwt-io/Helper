# Personal AI Agent

A 24x7 personal AI assistant running on your local Windows machine, accessible via a web UI. Built with LangGraph + LangChain, powered by OpenRouter or Ollama, with filesystem tools via Model Context Protocol (MCP), computer control (PyAutoGUI), persistent memory (SQLite + ChromaDB), a daily cost cap, and auto-restart via PM2.

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
13. [Cost Management](#cost-management)
14. [Extending the Agent](#extending-the-agent)

---

## Overview

The agent runs as a FastAPI web server on `http://127.0.0.1:8000` (localhost only, no auth). You chat with it through a browser UI. It can read/write files in your workspace, control your computer via PyAutoGUI (click, type, scroll, hotkeys), and more.

**Trigger model:** On-message only. The agent does nothing unless you send it a message. (Future versions may add scheduled/autonomous tasks.)

**Safety:** A hard daily cost cap ($1/day by default) stops the agent from spending beyond a limit. The persona instructs it to confirm before destructive actions.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.12+ | Core runtime |
| Agent framework | LangGraph 1.2 + LangChain 1.3 | Graph-based agent loop |
| LLM provider | OpenRouter **or** Ollama | Cloud (pay-per-token) or local (free) |
| LLM client | `langchain-openai` / `langchain-ollama` | Swap backend via `LLM_BACKEND` env var |
| Tool protocol | MCP (Model Context Protocol) | Standard tool interface |
| MCP servers | `@modelcontextprotocol/server-filesystem` | File tools (Node/npx) |
| Memory (history) | SQLite via SQLAlchemy | Per-session chat log + daily cost ledger |
| Memory (recall) | ChromaDB (persistent, local) | Vector search of past turns for context recall |
| Web server | FastAPI + Uvicorn | Async HTTP + WebSocket |
| Web UI | Vanilla HTML/JS (single page) | Chat interface, no build step |
| Process manager | PM2 (Node.js) | Auto-restart, keep-alive, auto-start on login |
| Config | YAML + `.env` | Settings + secrets |

---

## Project Structure

```
C:\Personal ai agent\
├── .env                  # secrets + backend config — NOT committed
├── .env.example          # template for .env
├── .gitignore
├── config.yaml           # all agent, MCP, memory, web settings
├── ecosystem.config.js   # PM2 process config
├── pm2_startup.vbs        # auto-start PM2 on Windows login (copied to Startup folder)
├── run_server.py          # launcher: python run_server.py → uvicorn
├── requirements.txt
├── .venv\                 # Python virtual environment
├── data\                  # runtime data (gitignored)
│   ├── history.db          # SQLite chat + cost
│   ├── chroma\             # ChromaDB vector store
│   ├── boot.log / boot.err # boot test logs
├── pm2_logs\              # PM2 stdout/stderr
├── agent\                 # main package
│   ├── __init__.py
│   ├── persona.md          # system prompt / personality
│   ├── mcp_manager.py     # spawn MCP servers, manage sessions + tool calls
│   ├── mcp_adapter.py     # convert MCP tools → LangChain StructuredTools
│   ├── graph.py           # LangGraph build + chat() entrypoint
│   ├── main.py            # FastAPI app: /health, /chat WebSocket, /, lifespan
│   ├── tools/              # custom non-MCP tools (empty for now)
│   ├── memory/
│   │   ├── db.py           # ChatDB: SQLite history + cost ledger
│   │   └── vector.py       # VectorStore: ChromaDB recall
│   └── web/
│       └── index.html      # chat UI
└── tests/
    ├── test_mcp_integration.py   # start MCP, list tools, call one
    └── test_graph.py             # build graph, one chat turn (needs API key)
```

---

## Architecture

### Flow: user sends a message

```
Browser (HTML/JS)
  → WebSocket /chat (FastAPI, main.py)
    → AgentGraph.chat() (graph.py)
        │
        ├─ 0. Cost guard: if spent_today >= $1 → refuse immediately
        ├─ 1. VectorStore.query() → relevant past turns (ChromaDB, cosine <0.6)
        ├─ 2. ChatDB.get_history() → last 20 messages (SQLite)
        ├─ 3. Assemble: [persona] + [recalled context] + [history] + [new msg]
        ├─ 4. Graph invoke (LangGraph):
        │       agent node (LLM call, bind_tools)
        │         ↓ route: tool_calls? → tools node → agent → ... → END
        │       (max 15 iterations, stop on cap hit)
        ├─ 5. Extract final AIMessage text
        ├─ 6. Persist: SQLite (user+assistant msg + cost) + ChromaDB (vector upsert)
        └─ 7. Return {text, cost_spent, messages}
    → WebSocket sends {type:"answer", text, cost_spent, spent_today, daily_cap}
  → UI renders assistant bubble + updates status bar
```

### LangGraph structure

```
State: {messages, cost_spent, session_id, tool_calls_made}

      ┌──────────┐
      │  START   │
      └────┬─────┘
           ↓
      ┌──────────┐  route()
      │  agent   │ ──────────→ END   (no tool_calls, or cap hit, or max iter)
      └────┬─────┘
           ↓ tool_calls present
      ┌──────────┐
      │  tools   │  (ToolNode dispatches to MCP servers via StructuredTool adapters)
      └────┬─────┘
           ↓
           └──→ agent  (loop back for next LLM turn with tool results)
```

### MCP integration

`MCPManager` (mcp_manager.py) spawns each MCP server as a subprocess via stdio:
- `npx -y @modelcontextprotocol/server-filesystem "C:\Personal ai agent"` → 14 file tools
- `npx -y @modelcontextprotocol/server-filesystem` → file read/write tools

Each server runs a `ClientSession` (MCP Python SDK). `MCPManager.list_tools_async()` collects all tools; `call_tool(name, args)` routes to the owning server.

`mcp_adapter.py` wraps each MCP tool as a `langchain_core.tools.StructuredTool` with:
- Auto-generated Pydantic args model (from the MCP tool's JSON schema)
- Async coroutine that calls `MCPManager.call_tool` and stringifies the result

These StructuredTools feed into `llm.bind_tools(tools)` and `ToolNode(tools)`, so LangGraph's prebuilt ToolNode handles dispatch automatically — no custom tool router needed.

### Memory

**SQLite (`ChatDB`, db.py):**
- `messages` table: (id, session_id, role, content, ts) — full chat log per session
- `cost_log` table: (id, date YYYY-MM-DD UTC, spent micro-USD) — daily ledger
- `get_history(session_id, limit=20)` → last N messages for context
- `add_cost(micro_usd)` / `spent_today()` → cost tracking

**ChromaDB (`VectorStore`, vector.py):**
- Persistent client at `data/chroma`
- Each completed turn (user+assistant) is stored as a document with session_id metadata
- `query(text, top_k=3)` → cosine similarity, filtered by distance < 0.6 before injecting as context
- Purpose: long-term recall across sessions (semantic memory)

### Cost model

- Tracked per LLM call via `response.usage_metadata.total_tokens * 0.000002` (rough $2/M-token estimate)
- Stored as integer micro-USD in SQLite `cost_log`
- `spent_today()` sums current UTC date
- Hard stop at `daily_cost_usd` (config, default 1.0) — checked both at `chat()` entry AND inside the graph's `route()` function
- When hit: agent returns a refusal message, no LLM call made, cap resets at UTC midnight

---

## Configuration

### `.env` (secrets — not committed)

```
# Backend selection: openrouter | ollama
LLM_BACKEND=ollama

# OpenRouter (used when LLM_BACKEND=openrouter)
OPENROUTER_API_KEY=sk-or-v1-REPLACE_WITH_YOUR_KEY
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=z-ai/glm-4.5

# Ollama (used when LLM_BACKEND=ollama)
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434

DAILY_COST_USD=1.0
WEB_PORT=8000
PERSONA_PATH=agent/persona.md
DATA_DIR=data
```

Get your OpenRouter key at https://openrouter.ai/ → sign up → create key.

### LLM Backend

The agent supports two backends, selected by `LLM_BACKEND` in `.env`:

| Backend | When to use | Cost | Setup |
|---------|-------------|------|-------|
| `ollama` | Local, private, no internet | Free | Install [Ollama](https://ollama.com), `ollama pull gemma4:e2b` |
| `openrouter` | Best model quality, cloud | Pay-per-token | Get API key from openrouter.ai |

**Switching backends:** Edit `LLM_BACKEND` in `.env`, then `pm2 restart ai-agent`. No code changes needed.

**Ollama setup:**
```powershell
# Install from https://ollama.com, then:
ollama pull qwen2.5:3b       # 3B model, ~2.4GB, fits 100% in 4GB VRAM
# Verify:
ollama list                   # should show qwen2.5:3b
# Start the Ollama service (runs on port 11434):
ollama serve
```

**GPU support:** Ollama auto-detects NVIDIA GPUs. Verify with `ollama ps` — the `PROCESSOR` column should show `100% GPU`. If it shows CPU/GPU split, the model is too large for your VRAM; use a smaller model. The default `qwen2.5:3b` (2.4GB) fits entirely in 4GB VRAM.

**Note on cost:** When using Ollama, `cost_spent` is always 0.0 (local model = free). The daily cost cap is still enforced but will never trigger.

**Note on GPU:** Ollama uses your NVIDIA GPU automatically. Check with `ollama ps` (look for `100% GPU` in the PROCESSOR column). If you see a CPU/GPU split, the model doesn't fit in your VRAM — switch to a smaller model (e.g. `qwen2.5:3b` for 4GB VRAM). Response time on GPU is ~10s vs ~108s on CPU.

### `config.yaml` (settings)

```yaml
llm:
  backend: ${LLM_BACKEND}          # openrouter | ollama
  openrouter:
    model: ${MODEL}
    api_key: ${OPENROUTER_API_KEY}
    base_url: ${OPENROUTER_BASE_URL}
  ollama:
    model: ${OLLAMA_MODEL}
    base_url: ${OLLAMA_BASE_URL}

daily_cost_usd: ${DAILY_COST_USD}

mcp:
  filesystem:                  # npx server, args list

memory:
  sqlite: ${DATA_DIR}/history.db
  chroma:  ${DATA_DIR}/chroma

web:
  port: ${WEB_PORT}
  host: 127.0.0.1             # localhost only
  auth: false                # no login

agent:
  persona_path: ${PERSONA_PATH}
  temperature: 0.3           # LLM sampling
  max_iterations: 25        # max tool-call loops per turn
```

Env vars (`${VAR}`) are expanded at load time by `_expand()` in graph.py:52.

### `agent/persona.md`

Edit this to change the agent's behavior, tone, and rules. It's injected as the first SystemMessage in every conversation. Current rules:
- Concise, direct, no filler
- Use tools when helpful
- Confirm before destructive actions
- Track context; recall past interactions
- Respect cost cap, never expose secrets

---

## How to Start

### First-time setup (already done — for reference)

```powershell
cd "C:\Personal ai agent"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env: set LLM_BACKEND=ollama or openrouter
# If ollama: ollama pull gemma4:e2b
# If openrouter: put your OpenRouter API key
```

### Starting the agent (PM2 — recommended)

```powershell
cd "C:\Personal ai agent"
pm2 start ecosystem.config.js
```

This launches `run_server.py` with the venv's Python, auto-restart on crash, logs to `pm2_logs/`. Takes ~15s to boot (MCP server spawn + tool listing).

Verify:
```powershell
pm2 ls                       # status should be "online"
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
# → {"status":"ok","tools":37,"spent_today":0.0,"daily_cap":1.0}
```

Open browser: `http://127.0.0.1:8000`

### Starting without PM2 (manual)

```powershell
cd "C:\Personal ai agent"
.\.venv\Scripts\python.exe run_server.py
```
Stops when you close the terminal. Use only for debugging.

### Auto-start on Windows login (already set up)

`pm2_startup.vbs` was copied to `C:\Users\<you>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\`. On every login it runs `pm2 resurrect` (silently restores the saved process list). `pm2 save` must have been run at least once for this to work — it was.

To re-trigger after changing config:
```powershell
pm2 delete ai-agent
pm2 start ecosystem.config.js
pm2 save                    # update the saved snapshot
```

---

## How to Use

1. Open `http://127.0.0.1:8000` in any browser.
2. Type a message, hit Enter or click Send.
3. You'll see "agent thinking..." while it runs.
4. The assistant's reply appears in a bubble.
5. Status bar (top-right) shows: `● online · 37 tools · $0.0000/1.0`

### Example prompts

- **"List the files in the current directory"** → uses `list_directory` filesystem tool
- **"Read the contents of README.md"** → uses `read_text_file`
- **"Create a new folder called `scratch`"** → uses `create_directory`
- **"Open Notepad and type a message"** → uses computer control (click, type_text)
- **"What did we talk about earlier?"** → recall pulls past turns from ChromaDB

### Tool categories available

**Filesystem (14 tools):** `read_file`, `read_text_file`, `read_media_file`, `read_multiple_files`, `write_file`, `edit_file`, `create_directory`, `list_directory`, `list_directory_with_sizes`, `directory_tree`, `move_file`, `search_files`, `get_file_info`, `list_allowed_directories`

- **Destructive actions:** per persona, the agent should confirm before deleting/overwriting. Watch the chat for confirmation prompts.
- **Multi-turn context:** last 20 SQLite messages + up to 3 vector-recalled past turns are sent as context each turn. Long conversations naturally include recent context.

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
Process is removed from the PM2 list. `pm2 save` to update the snapshot (otherwise `pm2 resurrect` on next login will try to restore a dead process — harmless, it just fails).

### Stop everything (PM2 daemon itself)

```powershell
pm2 kill
```
Kills the PM2 daemon and all managed processes. PM2 stays installed; just not running. Next `pm2 start` will re-spawn the daemon.

### Disable auto-start on login

Delete the startup file:
```powershell
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\pm2_startup.vbs"
```

### Stop the manual (non-PM2) run

Press `Ctrl+C` in the terminal running `run_server.py`.

---

## How to Restart

### After changing config.yaml or .env

```powershell
cd "C:\Personal ai agent"
pm2 restart ai-agent
```
PM2 kills the process and spawns a fresh one. Takes ~15s to boot MCP servers.

### After changing code (Python files)

Same — `pm2 restart ai-agent`. PM2 does NOT watch files by default (`watch: false` in ecosystem.config.js) to avoid restart loops during edits.

### Hard reset (clear memory)

To wipe all chat history and vector memory:
```powershell
pm2 stop ai-agent
Remove-Item data\history.db -Force -ErrorAction SilentlyContinue
Remove-Item data\chroma -Recurse -Force -ErrorAction SilentlyContinue
pm2 start ai-agent
```
Fresh SQLite + ChromaDB will be created on next boot.

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
- `AgentGraph ready. tools=14` — graph compiled, server accepting connections
- `Application startup complete.` — FastAPI lifespan done, /health is live
- `127.0.0.1:xxxx - "GET /health HTTP/1.1" 200 OK` — health check hit
- `127.0.0.1:xxxx - "GET /chat WebSocket"` — WS connection
- `WS connected` / `WS disconnected` — chat session lifecycle
- `chat error` — exception during a turn (look at the following traceback)

---

## Understanding the Code

### `agent/mcp_manager.py` — MCPManager

Spawns and talks to MCP servers. Key methods:
- `start()` — async. Reads `config.yaml` → for each MCP server, builds `StdioServerParameters`, calls `stdio_client(params)` from MCP SDK, opens a `ClientSession`, calls `session.initialize()`, calls `session.list_tools()` to register tools. All sessions held in `self._sessions` dict, tool→server map in `self._tools`.
- `list_tools_async()` — re-fetches tools from all live sessions. Returns `mcp.types.Tool` objects.
- `call_tool(name, args)` — looks up which server owns the tool, calls `session.call_tool(name, arguments=args)` on that session. Returns MCP `CallToolResult` with `.content` (list of `TextContent` etc.) and `.isError`.
- `stop()` — closes the `AsyncExitStack`, which tears down all stdio transports (subprocesses are killed).

### `agent/mcp_adapter.py` — MCP → LangChain bridge

- `_schema_to_pydantic(name, schema)` — reads MCP tool's `inputSchema` (JSON Schema), builds a Pydantic `BaseModel` with the right field types and required/optional flags. This becomes the tool's `args_schema` so LangChain's ToolNode can validate inputs.
- `_make_tool(mcp_tool, mgr)` — creates a `StructuredTool.from_function` with:
  - `coroutine` = async fn that calls `mgr.call_tool(name, kwargs)` and joins the text content
  - `name` = MCP tool name (preserved, so LLM sees the real name)
  - `description` = MCP tool description
  - `args_schema` = the generated Pydantic model
- `build_langchain_tools(mgr)` — async. Fetches all MCP tools via `mgr.list_tools_async()`, wraps each. Returns a `list[StructuredTool]` ready for `llm.bind_tools()`.

This is the integration layer: MCP tools become LangChain tools, which become LangGraph-dispatchable via the prebuilt `ToolNode`.

### `agent/graph.py` — AgentGraph

The brain. Key parts:

**`AgentState` (TypedDict):** the graph state schema. Fields: `messages`, `cost_spent`, `session_id`, `tool_calls_made`. LangGraph merges node returns into this state between nodes.

**`setup()` (async):** the only entrypoint to initialize everything.
1. `_load()` — reads config.yaml, expands env vars recursively, loads persona.md, instantiates `ChatDB` + `VectorStore`.
2. `mcp.start()` — spawns MCP servers.
3. `build_langchain_tools(mcp)` — wraps MCP tools as LangChain tools.
4. Creates `ChatOpenAI` pointed at OpenRouter (uses `openai_api_base` to redirect to OpenRouter's API; OpenRouter is OpenAI-compatible).
5. `llm.bind_tools(tools)` — gives the LLM the tool-calling interface.
6. `ToolNode(tools)` — LangGraph's prebuilt tool dispatcher. When the LLM emits `tool_calls`, the ToolNode routes each call to the matching StructuredTool, runs its coroutine (which calls MCP), and returns `ToolMessage`s with results.
7. `_build_graph()` — wires nodes + edges.

**`_build_graph()`:** defines three things:
- `agent_node` — injects persona as SystemMessage if not present, calls `llm_with_tools.ainvoke(msgs)`, extracts token usage to update `cost_spent`.
- `tools_node` — delegates to ToolNode (which calls MCP via the StructuredTool adapters), increments `tool_calls_made`.
- `route()` — conditional edge from agent. Returns `"tools"` if the last AIMessage has `tool_calls`, else `END`. Also returns `END` if cost cap hit or `max_iterations` reached — these are the loop termination guards.

Edges: `START → agent`, `agent →(route)→ tools | END`, `tools → agent` (loop back). Compiled with `g.compile()`.

**`chat(user_text, session_id)` (async):** the user-facing API. One call = one turn. Steps:
0. **Cost guard** — if `spent_today() >= daily_cap`, return refusal immediately. No LLM call.
1. **Vector recall** — `VectorStore.query(user_text, top_k=3)`, filter by cosine distance < 0.6 (only genuinely similar past turns).
2. **History load** — `ChatDB.get_history(session_id, limit=20)` → recent messages as LangChain message objects.
3. **Assemble** — `[SystemMessage(recalled context if any)] + prior history + [HumanMessage(user_text)]`.
4. **Graph invoke** — `self._graph.ainvoke(state)`. LangGraph runs agent→tools→agent→... until route returns END.
5. **Extract final** — scan `out_msgs` in reverse for the last AIMessage without `tool_calls` (that's the final answer to the user).
6. **Persist** — save user msg + assistant msg to SQLite, add cost to daily ledger, upsert the turn to ChromaDB for future recall.
7. **Return** — `{text, cost_spent, messages}`.

**`close()`** — calls `mcp.stop()` to tear down MCP subprocesses. Called by FastAPI lifespan on shutdown.

### `agent/main.py` — FastAPI server

- `lifespan` context manager — runs at app startup/shutdown. Creates `AgentGraph`, calls `setup()` (spawns MCP, builds graph), stores in module-global `_graph`. On shutdown calls `_graph.close()`.
- `GET /` — serves `agent/web/index.html`.
- `GET /health` — JSON status: tools count, spent_today, daily_cap. Useful for monitoring.
- `WebSocket /chat` — the chat endpoint. Accepts connection, loops: receive JSON `{text, session_id}` → send `{type:"thinking"}` → call `_graph.chat()` with 180s timeout → send `{type:"answer", text, cost_spent, spent_today, daily_cap}`. On error sends `{type:"error", text}`. On disconnect, logs and returns.
- `main()` — runs uvicorn on host/port from env. Use `python run_server.py` which imports and calls this.

### `agent/memory/db.py` — ChatDB

SQLAlchemy ORM over SQLite. Two tables:
- `messages`: (id PK, session_id idx, role, content, ts default UTC now). `add_message` inserts; `get_history` returns last N by id desc, then reverses to chronological.
- `cost_log`: (id PK, date idx "YYYY-MM-DD" UTC, spent int micro-USD). `add_cost(micro_usd)` either updates today's row or creates it. `spent_today()` reads today's row, divides by 1e6 to USD.

Why micro-USD as int? Avoids float drift over many small additions.

### `agent/memory/vector.py` — VectorStore

Thin wrapper over `chromadb.PersistentClient`. Single collection `chat`, cosine space.
- `add(ids, texts, metas)` — upsert. Each completed turn stored as `"User: ...\nAssistant: ..."` with `session_id` metadata.
- `query(text, top_k)` — `collection.query(query_texts=[text], n_results=top_k)`. Returns docs, metas, distances. Filtered in `graph.py` by `distance < 0.6` (cosine distance, lower = more similar; 0.6 ≈ 80% similar).
- `count()` — for empty-store guard before querying (ChromaDB errors on empty queries).

### `agent/web/index.html` — chat UI

Single HTML file, no build step. Dark theme, mobile-friendly layout. Vanilla JS WebSocket client:
- Connects to `ws://${location.host}/chat` on load.
- `send()` — sends `{text, session_id:"default"}`, disables input while busy.
- On `{type:"thinking"}` — shows italic "agent thinking..." placeholder.
- On `{type:"answer"}` — removes thinking placeholder, adds assistant bubble, updates status bar with cost.
- On `{type:"error"}` — red error bubble.
- Status bar fetched from `/health` on connect; updated after each answer.

### `ecosystem.config.js` — PM2 config

 Tells PM2 how to run the agent:
- `script: run_server.py`, `interpreter: .venv/Scripts/python.exe` — uses the venv Python, not system Python.
- `autorestart: true`, `max_restarts: 10`, `restart_delay: 3000` — if the process crashes, PM2 waits 3s and respawns, up to 10 times. After 10 rapid crashes PM2 stops (avoid crash loops).
- `max_memory_restart: 500M` — if RSS exceeds 500MB, PM2 restarts (memory leak guard).
- `watch: false` — no file watching. Manual restart only.
- Logs to `pm2_logs/agent.{out,err}.log`, timestamped.

### `pm2_startup.vbs`

Windows can't easily run `pm2 startup` (needs admin). This VBS script is a workaround: it runs `cmd /c pm2 resurrect` silently (window hidden via `Run ..., 0, False`). Copied to the user's `Startup` folder, so Windows runs it on every login. `pm2 resurrect` reads the saved process list (`~/.pm2/dump.pm2`, created by `pm2 save`) and restarts all saved apps.

---

## Troubleshooting

### `GET /health` returns "Unable to connect"

- `pm2 ls` — is `ai-agent` `online`? If `errored`, check `pm2 logs ai-agent --err`.
- If just `stopped`, run `pm2 start ai-agent`.
- If boot is slow (MCP servers take ~15s to spawn on first run), wait and retry.

### MCP server fails to start

- **filesystem:** check `npx -y @modelcontextprotocol/server-filesystem --help` works. Node must be installed.
- **"Connection closed"** in logs — the MCP server subprocess crashed. Check that the npm package name in `config.yaml` is exactly right.

### Chat returns "Daily cost cap reached"

- You've hit $1 today. Resolves at UTC midnight. To reset manually, clear the cost row:
  ```powershell
  .\.venv\Scripts\python.exe -c "from agent.memory.db import ChatDB; d=ChatDB(); import sqlalchemy; s=d.Session(); s.query(d.__import__('sqlalchemy').text('DELETE FROM cost_log WHERE date = :d'), d.today_str()); s.commit()"
  ```
  Or just delete `data/history.db` and restart (wipes chat history too).
- To raise the cap: edit `DAILY_COST_USD` in `.env` or `daily_cost_usd` in `config.yaml`, then `pm2 restart ai-agent`.

### Chat returns empty text

- The agent made only tool calls, never a final AIMessage without tool_calls. Shouldn't happen (route sends to END after max_iterations). Check `pm2 logs` for the last turn. If it's a tool result that should have been followed by an LLM call, the graph terminated early — check cost cap.

### "OPENROUTER_API_KEY" error on boot

- Only applies when `LLM_BACKEND=openrouter`. Check `.env` has `LLM_BACKEND=openrouter` and a valid `OPENROUTER_API_KEY`. If using Ollama, ensure `LLM_BACKEND=ollama`.

### Ollama errors / connection refused

- Verify Ollama is running: `ollama serve` (or check `http://127.0.0.1:11434/api/tags` in browser).
- Verify model is pulled: `ollama list` should show `gemma4:e2b` (or your configured model).
- Check `OLLAMA_BASE_URL` in `.env` matches `http://127.0.0.1:11434`.
- If Ollama is slow (CPU-only), increase timeout patience — 5.1B model can take 30-120s per turn.

### ChromaDB error on Windows

- Needs Microsoft VC++ runtime. Install "Visual C++ Redistributable for Visual Studio 2015-2022" from Microsoft. Reboot. Retry.

### PM2 says "App not found" on resurrect

- You need `pm2 save` after `pm2 start`. Run `pm2 start ecosystem.config.js && pm2 save`.

---

## Cost Management

- **Default cap:** $1.00/day (UTC)
- **Reset:** automatic at UTC midnight
- **Where stored:** `data/history.db`, table `cost_log`, column `spent` (micro-USD integer)
- **Tracking granularity:** per LLM call. Formula: `total_tokens * 0.000002` (approximate; GLM-5.2 is far cheaper, this is a conservative estimate so the cap triggers before you overspend).
- **Enforcement points:** two — at `chat()` entry (refuses before any LLM call) and in `route()` mid-graph (stops the tool-call loop).
- **Status visibility:** `/health` endpoint returns `spent_today` + `daily_cap`. The web UI status bar shows `$/cap` after every turn.
- **Tuning:** edit `DAILY_COST_USD` in `.env`, restart. Or `daily_cost_usd` in `config.yaml` (same value, env-expanded).

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
Add `GITHUB_TOKEN=...` to `.env`. `MCPManager._parse_servers` will pick it up; `build_langchain_tools` will wrap its tools; the LLM gets them via `bind_tools`. Restart: `pm2 restart ai-agent`.

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
The ToolNode will dispatch it like any MCP tool. Restart.

### Change the persona

Edit `agent/persona.md`. `pm2 restart ai-agent`. Takes effect on next `chat()` call (persona is read at `setup()` time, not per-message).

### Change the model

**Ollama:** `ollama pull <model>`, then edit `.env`: `OLLAMA_MODEL=<model>`, `pm2 restart ai-agent`. Browse models at https://ollama.com/library. Make sure the model supports tool-calling (most recent models do — gemma4, llama3.1, qwen2.5, etc.).

**OpenRouter:** Edit `.env`: `MODEL=openrouter/anthropic/claude-3.5-sonnet` (or any OpenRouter model ID). Restart. No code changes — `ChatOpenAI` accepts any OpenAI-compatible model ID via OpenRouter.

### Switch between Ollama and OpenRouter

Edit `.env`: set `LLM_BACKEND=ollama` or `LLM_BACKEND=openrouter`. `pm2 restart ai-agent`. All config (model name, API key, base URL) is read from the corresponding section in `.env`.

### Add multi-session support

`ChatDB` already keys by `session_id`. The WebSocket handler in `main.py` reads `session_id` from the client JSON. To use multiple sessions, have the frontend send different `session_id` values (e.g. `session-1`, `session-2`). Each session has independent history. Vector recall is shared across sessions (intentional — cross-session memory).

### Add authentication

In `config.yaml` set `web.auth: true`. In `main.py`, add a dependency on the WebSocket that checks an `Authorization` header or query param `?token=...` against a value in `.env` (`WEB_TOKEN`). Reject if mismatched. For v1 we skipped this (localhost only).

### Move to a VPS for true 24x7

Local Windows machine sleeps → agent dies. For genuine 24x7:
1. Deploy to a Linux VPS (DigitalOcean / Hetzner / etc., $5-10/mo).
2. Install Node, Python, clone this repo, recreate `.venv`, `pip install -r requirements.txt`.
3. `pm2 start ecosystem.config.js && pm2 save`.
4. On Linux, `pm2 startup` works natively (creates a systemd service) — no VBS hack needed.
5. Point a domain at the VPS, add HTTPS via Caddy/nginx, optionally add auth.
6. Update `config.yaml` MCP filesystem path to the VPS workspace.

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
.\.venv\Scripts\python.exe run_server.py

# Auto-start (re-copy script after Windows profile reset)
Copy-Item pm2_startup.vbs "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\"

# Disable auto-start
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\pm2_startup.vbs"
```
