"""Lightweight 1-hop import resolution.

Maps a source file to the local workspace files it imports. Used to pull in
structural neighbors (callers/callees live nearby) without indexing the world.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from context_eng.workspace import read_text

_TS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_TS_INDEX = tuple(f"index{ext}" for ext in _TS_EXTS)

_JS_IMPORT_RE = re.compile(
    r"""(?:import\s[^'"]*from\s*|import\s*|require\s*\(\s*)['"]([^'"]+)['"]""",
)


def _python_imports(source: str, path: Path, workspace: Path) -> list[Path]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: list[Path] = []

    def module_to_path(module: str, level: int) -> list[Path]:
        candidates: list[Path] = []
        if level > 0:
            base = path.parent
            for _ in range(level - 1):
                base = base.parent
            parts = module.split(".") if module else []
            target = base.joinpath(*parts)
        else:
            target = workspace.joinpath(*module.split("."))
        candidates.append(target.with_suffix(".py"))
        candidates.append(target / "__init__.py")
        return candidates

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.extend(module_to_path(alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            results.extend(module_to_path(module, node.level or 0))

    resolved: list[Path] = []
    for cand in results:
        if cand.is_file():
            resolved.append(cand.resolve())
    return resolved


def _resolve_js_specifier(spec: str, path: Path) -> Path | None:
    if not spec.startswith("."):
        return None  # bare/package import, not local
    base = (path.parent / spec).resolve()
    if base.is_file():
        return base
    for ext in _TS_EXTS:
        cand = base.with_suffix(ext)
        if cand.is_file():
            return cand
    if base.is_dir():
        for index in _TS_INDEX:
            cand = base / index
            if cand.is_file():
                return cand
    return None


def _js_imports(source: str, path: Path) -> list[Path]:
    resolved: list[Path] = []
    for spec in _JS_IMPORT_RE.findall(source):
        target = _resolve_js_specifier(spec, path)
        if target is not None:
            resolved.append(target)
    return resolved


def local_imports(path: Path, workspace: Path) -> list[Path]:
    """Return resolved local files imported by ``path`` (1 hop)."""
    path = path.resolve()
    workspace = workspace.resolve()
    source = read_text(path)
    if not source:
        return []

    if path.suffix in (".py", ".pyi"):
        found = _python_imports(source, path, workspace)
    elif path.suffix in _TS_EXTS:
        found = _js_imports(source, path)
    else:
        return []

    # Dedupe, keep only files inside the workspace, drop self.
    seen: dict[str, Path] = {}
    for f in found:
        try:
            f.relative_to(workspace)
        except ValueError:
            continue
        if f == path:
            continue
        seen[f.as_posix()] = f
    return list(seen.values())
