"""Workspace root resolution for multi-project MCP usage.

Resolution order (first match wins):
1. Explicit ``workspace_root`` tool argument
2. ``CONTEXT_ENG_WORKSPACE`` environment variable
3. Process current working directory (Cursor typically sets this to the open
   project root when launching MCP servers)
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_workspace(workspace_root: str | None = None) -> Path:
    """Return the absolute workspace path to index for a tool call."""
    if workspace_root:
        return Path(workspace_root).expanduser().resolve()
    env = os.environ.get("CONTEXT_ENG_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()
