"""Test simpler browser task: navigate directly to a URL and extract."""
import asyncio
import json
import os
import sys

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8000/chat") as ws:
        await ws.send(json.dumps({
            "text": "Use browser_navigate to go to https://en.wikipedia.org/wiki/Python_(programming_language) and tell me the first paragraph of the article.",
            "session_id": "test-wiki",
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
