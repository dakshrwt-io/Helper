"""Test web search via browser: navigate to Google, search, extract."""
import asyncio
import json
import os
import sys

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/chat") as ws:
        await ws.send(json.dumps({
            "text": "Use the browser to search Google for 'latest Python version 2026' and tell me what you find. Navigate to google.com, type the search query, and read the results.",
            "session_id": "test-search",
        }))
        while True:
            try:
                data = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
            except asyncio.TimeoutError:
                print("TIMEOUT"); return
            print(f"[{data['type']}]")
            if data["type"] == "answer":
                print(f"TEXT: {data['text'][:800]}")
                print(f"cost: ${data['cost_spent']}, today: ${data['spent_today']}")
                return
            if data["type"] == "error":
                print(f"ERR: {data['text'][:800]}"); return


if __name__ == "__main__":
    asyncio.run(main())
