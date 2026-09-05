# Helper

Helper is a local AI assistant for Windows. It provides a password-protected browser chat interface and an optional Telegram bot, with tools for approved desktop control, files, MCP services, delegated sub-agents, and persistent local memory.

It is designed to run on your own machine. Messages are handled only when you send them; it does not run autonomous background tasks.

## What it can do

- Chat through the local web interface or Telegram.
- Use a selectable LLM backend: Ollama, OpenRouter, DeepSeek, or NVIDIA NIM.
- Work with files through the MCP filesystem server, scoped to a directory you choose.
- Delegate file, coding, Git, browser, desktop, and general tasks to specialized sub-agents.
- Remember conversations locally with SQLite and ChromaDB.
- Show live execution traces in the browser.
- Control the Windows desktop after a user-approved, time-limited lease.

## Security at a glance

This project can access files and, when enabled, operate the desktop. Treat it like any other locally running automation tool.

- The server defaults to `127.0.0.1`; do not expose it to a network without HTTPS, a strong `WEB_TOKEN`, and a deliberate origin policy.
- The web interface requires `WEB_TOKEN`. It will not start without one.
- Desktop actions require an active, browser-approved lease. The lease duration and action rate are configurable.
- Keep secrets in `.env`, never in `config.yaml` or source files. `.env` and common local-secret variants are ignored by Git.
- Set `FILESYSTEM_MCP_DIR` to the narrowest directory the assistant needs. MCP filesystem tools can modify files inside that directory.

## Requirements

- Windows 10 or later for desktop automation.
- Python 3.12 or later.
- Node.js 18 or later for the filesystem MCP server (`npx`).
- An LLM provider:
  - [Ollama](https://ollama.com/) for local models, or
  - an API key for OpenRouter, DeepSeek, or NVIDIA NIM.
- Optional: PM2 for keeping the service running after you have verified a manual start.

## Quick start

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
Copy-Item .env.example .env
```

Edit `.env` before starting. At minimum, set a strong web token, a filesystem root, and an LLM backend.

```dotenv
# Required: use a long, private random value.
WEB_TOKEN=replace-this-with-a-long-random-value

# Restrict filesystem tools to this directory.
FILESYSTEM_MCP_DIR=C:\path\you\want\the\agent\to\access

# Choose one backend.
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

For a local Ollama setup, install Ollama and pull the selected model:

```powershell
ollama pull qwen2.5:3b
```

Start Helper:

```powershell
python run_server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), enter `WEB_TOKEN`, and begin a chat. Check the service with:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

## Configuration

`.env.example` is the complete, safe template. Copy it to `.env`; do not commit the copy.

| Setting | Purpose |
|---|---|
| `LLM_BACKEND` | `ollama`, `openrouter`, `deepseek`, or `nvidia` |
| `WEB_TOKEN` | Required password for the web interface and Bearer-token API access |
| `WEB_PORT` / `WEB_HOST` | Server address; defaults to `8000` and `127.0.0.1` |
| `WEB_ALLOWED_ORIGINS` | Comma-separated extra origins permitted to open the WebSocket |
| `WEB_COOKIE_SECURE` | Set to `true` when serving over HTTPS |
| `FILESYSTEM_MCP_DIR` | Directory exposed to filesystem MCP tools |
| `DATA_DIR` | Local SQLite and ChromaDB storage directory |
| `TELEGRAM_ENABLED` | Enables the optional Telegram bot when `true` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token; required only when Telegram is enabled |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated Telegram user IDs allowed to chat |
| `COMPUTER_CONTROL_ENABLED` | Enables desktop-control tools |
| `COMPUTER_CONTROL_LEASE_SECONDS` | Maximum desktop-approval lease length, capped at 300 seconds |
| `COMPUTER_CONTROL_RATE_LIMIT` | Maximum desktop actions per minute per session |

`config.yaml` contains non-secret application settings and resolves `${VARIABLE}` entries from `.env`. It also defines MCP servers, memory locations, tool limits, and sub-agent roles. Restart the server after changing either file.

### Cloud providers

Set `LLM_BACKEND` and the matching variables in `.env`:

| Backend | Required settings |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY`, `MODEL`, `OPENROUTER_BASE_URL` |
| DeepSeek | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL` |
| NVIDIA NIM | `NVIDIA_API_KEY`, `NVIDIA_MODEL` |
| Ollama | `OLLAMA_MODEL`, `OLLAMA_BASE_URL` |

`GROQ_API_KEY` and `VISION_API_KEY` are optional. When vision is configured, screenshots can be analyzed before the acting model continues.

## Running with PM2

Run Helper manually first. The supplied `ecosystem.config.js` contains machine-specific `cwd` and Python paths, so update those values to match your checkout and virtual environment before using PM2.

```powershell
npm install --global pm2
pm2 start ecosystem.config.js
pm2 save
pm2 ls
pm2 logs ai-agent
```

Useful commands:

```powershell
pm2 restart ai-agent   # apply configuration or code changes
pm2 stop ai-agent      # stop without removing the process definition
pm2 delete ai-agent    # remove the process definition
pm2 logs ai-agent --err --lines 50
```

`pm2_startup.vbs` can be copied to the Windows Startup folder after `pm2 save` if you want PM2 to restore its process list when you sign in.

## Optional Telegram bot

Set these values in `.env` and restart Helper:

```dotenv
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_ALLOWED_USERS=123456789
```

The bot rejects users who are not in `TELEGRAM_ALLOWED_USERS`. Keep the bot token private.

## Project layout

```text
agent/
  graph.py              Agent graph and chat entry point
  main.py               FastAPI app, web authentication, and WebSocket chat
  config_loader.py      Configuration, persona, and memory initialization
  llm_factory.py        LLM backend construction
  mcp_manager.py        MCP process, tool, and health management
  memory/               SQLite history and ChromaDB vector memory
  subagents/            Delegated task graphs and task tool
  tools/computer.py     Desktop-control tools and lease enforcement
  web/index.html        Browser interface
  chat_providers/       Telegram integration
config.yaml             Non-secret runtime configuration
run_server.py           Starts the agent, optional Telegram bot, and web server
requirements.txt        Python dependencies
ecosystem.config.js     PM2 configuration
tests/                  Automated test suite
```

## Development

Run the test suite:

```powershell
python -m pytest -q
```

Runtime data is stored under `DATA_DIR` (default: `data/`) and is ignored by Git. To clear local conversation history, stop the service and remove the SQLite database and ChromaDB directory under that location.

## Troubleshooting

| Problem | Check |
|---|---|
| Server refuses to start | Set a non-empty `WEB_TOKEN` in `.env`. |
| Browser cannot connect | Confirm the service is running, use `http://127.0.0.1:8000`, and check `WEB_ALLOWED_ORIGINS` if using a reverse proxy. |
| Ollama requests fail | Run `ollama list`, verify the selected model is installed, and confirm `OLLAMA_BASE_URL`. |
| Filesystem tools are unavailable | Install Node.js, verify `npx` is available, and confirm `FILESYSTEM_MCP_DIR` exists. |
| Desktop action is denied | Enable `COMPUTER_CONTROL_ENABLED` and approve a new desktop lease in the web interface. |
| Telegram does not start | Set `TELEGRAM_ENABLED=true`, provide a token, and check `TELEGRAM_ALLOWED_USERS`. |
| Need logs | Run `pm2 logs ai-agent --err --lines 50`, or inspect console output from `python run_server.py`. |

