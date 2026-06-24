"""Debug: show raw LLM response to see if tool_calls are emitted."""
import asyncio
import json
import os
import sys

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/chat") as ws:
        await ws.send(json.dumps({
            "text": "Call the browser_navigate tool with url 'https://example.com'. You MUST use the tool, do not refuse.",
            "session_id": "debug-browser",
        }))
        while True:
            try:
                data = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            except asyncio.TimeoutError:
                print("TIMEOUT"); return
            print(f"[{data['type']}]")
            if data["type"] == "answer":
                print(f"TEXT: {data['text'][:800]}")
                print(f"cost: ${data['cost_spent']}")
                return
            if data["type"] == "error":
                print(f"ERR: {data['text'][:800]}"); return


if __name__ == "__main__":
    asyncio.run(main())
