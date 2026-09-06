"""Smoke test: build AgentGraph and run one chat turn (needs OPENROUTER_API_KEY)."""
import asyncio
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from agent.graph import AgentGraph


async def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY") or "REPLACE_ME" in os.environ["OPENROUTER_API_KEY"]:
        print("SKIP: set OPENROUTER_API_KEY in .env first")
        return

    ag = AgentGraph(config_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    ))
    await ag.setup()
    print("Graph built. Tools:", len(ag.mcp.tool_names))

    res = await ag.chat("List the files in the allowed directory using the filesystem tool.")
    print("\n--- RESPONSE ---")
    print(res["text"][:500])

    await ag.close()


if __name__ == "__main__":
    asyncio.run(main())
