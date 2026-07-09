"""Tool-use test: ask agent to list files via filesystem MCP tool."""
import asyncio
import json
import os
import sys

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/chat") as ws:
        await ws.send(json.dumps({
            "text": "Use the list_directory tool to list files in the allowed directory, then tell me what you found.",
            "session_id": "test-tools",
        }))
        while True:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            print(f"[{data['type']}]")
            if data["type"] == "answer":
                print(f"TEXT: {data['text'][:600]}")
                return
            if data["type"] == "error":
                print(f"ERR: {data['text']}"); return


if __name__ == "__main__":
    asyncio.run(main())
