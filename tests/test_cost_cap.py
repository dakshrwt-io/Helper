"""Cost cap test: inject fake spend, verify agent refuses."""
import asyncio
import json
import os
import sys

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory.db import ChatDB


async def main() -> None:
    # 1. inject fake cost = $1.5 (> cap)
    d = ChatDB()
    d.add_cost(1_500_000)  # micro-USD = $1.5
    print(f"Injected fake spend. spent_today = ${d.spent_today()}")

    # 2. send a chat, expect refusal
    async with websockets.connect("ws://127.0.0.1:8000/chat") as ws:
        await ws.send(json.dumps({"text": "Say hello", "session_id": "cap-test"}))
        while True:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            print(f"[{data['type']}]")
            if data["type"] == "answer":
                print(f"TEXT: {data['text']}")
                print("PASS" if "cap" in data["text"].lower() else "FAIL")
                return
            if data["type"] == "error":
                print(f"ERR: {data['text']}"); return


if __name__ == "__main__":
    asyncio.run(main())
