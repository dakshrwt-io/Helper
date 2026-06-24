"""End-to-end smoke test via WebSocket: send a chat, expect answer."""
import asyncio
import json
import os
import sys

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    url = "ws://127.0.0.1:8000/chat"
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"text": "Say 'pong' and nothing else.", "session_id": "test"}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            data = json.loads(raw)
            print(f"[{data['type']}]")
            if data["type"] == "answer":
                print(f"TEXT: {data['text']}")
                print(f"cost_spent: ${data['cost_spent']}")
                print(f"spent_today: ${data['spent_today']} / ${data['daily_cap']}")
                return
            if data["type"] == "error":
                print(f"ERROR: {data['text']}")
                return


if __name__ == "__main__":
    asyncio.run(main())
