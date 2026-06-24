"""Test that trace events are streamed via WebSocket during a chat turn."""
import asyncio
import json
import os
import sys

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    events: list[dict] = []
    async with websockets.connect("ws://127.0.0.1:8000/chat") as ws:
        await ws.send(json.dumps({"text": "Say 'pong' and nothing else.", "session_id": "test-trace-ws"}))
        while True:
            try:
                data = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            except asyncio.TimeoutError:
                print("TIMEOUT"); break

            if data["type"] == "trace":
                events.append(data["event"])
            elif data["type"] == "answer":
                print(f"answer: {data['text'][:100]}")
                break
            elif data["type"] == "error":
                print(f"error: {data['text']}"); break

    print(f"\nreceived {len(events)} trace events")
    types = [e["type"] for e in events]
    print(f"event types: {types}")

    expected = {"turn_started", "llm_started", "llm_completed", "turn_completed"}
    found = set(types) & expected
    print(f"expected events found: {found}")
    print("PASS" if expected.issubset(set(types)) else "FAIL")

    for e in events:
        if e["type"] == "llm_completed":
            print(f"  llm_completed content: {e.get('content', '')[:60]}")
            print(f"  llm_completed usage: {e.get('usage', {})}")
            break


if __name__ == "__main__":
    asyncio.run(main())
