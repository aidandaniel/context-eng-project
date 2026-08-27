"""Merge the context-eng MCP server into ~/.cursor/mcp.json.

Does not delete other mcpServers entries. Used by install.sh / install.ps1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: register_cursor_mcp.py <python-executable>", file=sys.stderr)
        return 2
    python = str(Path(sys.argv[1]).expanduser().resolve())
    cursor_dir = Path.home() / ".cursor"
    mcp_path = cursor_dir / "mcp.json"
    entry = {"command": python, "args": ["-m", "context_eng.server"]}

    data: dict = {"mcpServers": {}}
    if mcp_path.is_file():
        try:
            loaded = json.loads(mcp_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            print(
                f"Warning: {mcp_path} is not valid JSON; creating a new file",
                file=sys.stderr,
            )
            data = {"mcpServers": {}}
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        servers = {}
        data["mcpServers"] = servers
    servers["context-eng"] = entry
    cursor_dir.mkdir(parents=True, exist_ok=True)
    mcp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Registered context-eng in {mcp_path}")
    print(f"  command: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
