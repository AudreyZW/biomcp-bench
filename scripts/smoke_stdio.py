"""End-to-end check over the real stdio transport.

Unit tests cover logic without network; this one spawns the server as a
subprocess and speaks MCP to it, which is the only way to catch protocol-level
and packaging mistakes (a broken console script, an unserialisable return type).

    uv run python scripts/smoke_stdio.py
"""

from __future__ import annotations

import asyncio
import json

from mcp import ClientSession, StdioServerParameters, stdio_client


async def main() -> None:
    params = StdioServerParameters(command="uv", args=["run", "biomcp"], cwd=".")
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        print(
            f"initialize    server={init.server_info.name} "
            f"v{init.server_info.version} protocol={init.protocol_version}"
        )

        tools = (await session.list_tools()).tools
        print(f"tools/list    {len(tools)}: {[t.name for t in tools]}")
        assert len(tools) == 6, "six tools is the design claim"

        templates = (await session.list_resource_templates()).resource_templates
        print(f"resources     {[t.uri_template for t in templates]}")

        result = await session.call_tool("get_structure", {"pdb_id": "6LU7"})
        text = "".join(getattr(c, "text", "") for c in result.content)
        entry = json.loads(text)
        print(f"call_tool     get_structure(6LU7) -> {len(text)} chars")
        print(f"              {entry['pdb_id']} at {entry['resolution_a']} A")
        assert entry["pdb_id"] == "6LU7"

    print("\nOK")


if __name__ == "__main__":
    asyncio.run(main())
