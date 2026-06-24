"""Direct MCP test: start playwright MCP, call browser_navigate."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters, stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@playwright/mcp",
            "--headless",
            "--executable-path",
            r"C:\Users\daksh\AppData\Local\ms-playwright\chromium-1228\chrome-win64\chrome.exe",
        ],
    )
    print("starting playwright MCP...")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"tools: {len(tools.tools)}")

            print("\ncalling browser_navigate...")
            result = await session.call_tool("browser_navigate", {"url": "https://example.com"})
            print(f"isError: {result.isError}")
            for c in result.content:
                txt = getattr(c, "text", None)
                if txt:
                    print(f"content: {txt[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
