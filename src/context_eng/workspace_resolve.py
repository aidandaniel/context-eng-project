"""Workspace root resolution for multi-project MCP usage.

Resolution order (first match wins):
1. Explicit ``workspace_root`` tool argument
2. ``CONTEXT_ENG_WORKSPACE`` environment variable
3. Process current working directory (Cursor typically sets this to the open
   project root when launching MCP servers)

After resolution, the path must fall under an allowlisted root:
- ``CONTEXT_ENG_ALLOWED_ROOTS`` (``os.pathsep``-separated absolute/expanduser paths), or
- if unset/empty: only ``Path.cwd().resolve()``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _allowed_roots() -> list[Path]:
    raw = os.environ.get("CONTEXT_ENG_ALLOWED_ROOTS", "").strip()
    if not raw:
        return [Path.cwd().resolve()]
    roots: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(Path(part).expanduser().resolve())
    return roots or [Path.cwd().resolve()]


def _is_under_allowed(resolved: Path, roots: list[Path]) -> bool:
    for root in roots:
        if resolved == root:
            return True
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def resolve_workspace(workspace_root: str | None = None) -> Path:
    """Return the absolute workspace path to index for a tool call.

    Raises:
        PermissionError: if the resolved path is outside the allowlisted roots.
        FileNotFoundError: if the resolved path does not exist.
        NotADirectoryError: if the resolved path exists but is not a directory.
    """
    if workspace_root:
        resolved = Path(workspace_root).expanduser().resolve()
    else:
        env = os.environ.get("CONTEXT_ENG_WORKSPACE")
        if env:
            resolved = Path(env).expanduser().resolve()
        else:
            resolved = Path.cwd().resolve()

    roots = _allowed_roots()
    if not _is_under_allowed(resolved, roots):
        raise PermissionError(
            f"Workspace path {resolved} is outside the allowlist. "
            "Set CONTEXT_ENG_ALLOWED_ROOTS to permit additional roots "
            f"(current allowlist: {os.pathsep.join(str(r) for r in roots)})."
        )

    if not resolved.exists():
        raise FileNotFoundError(f"Workspace path does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Workspace path is not a directory: {resolved}")

    return resolved
