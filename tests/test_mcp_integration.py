"""Integration test: start MCP servers, list tools, call a filesystem tool."""
import asyncio
import os
import sys

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.mcp_manager import MCPManager


async def main() -> None:
    mgr = MCPManager(config_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    ))
    await mgr.start()

    print(f"\nServers online: {mgr.server_names}")
    print(f"Total tools: {len(mgr.tool_names)}")
    print(f"Tools: {mgr.tool_names[:20]}{'...' if len(mgr.tool_names) > 20 else ''}")

    # Try a safe filesystem call: list allowed dirs
    if "list_allowed_directories" in mgr.tool_names:
        res = await mgr.call_tool("list_allowed_directories", {})
        print(f"\nlist_allowed_directories result: {res}")

    await mgr.stop()


if __name__ == "__main__":
    asyncio.run(main())
