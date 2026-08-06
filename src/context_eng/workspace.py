"""Workspace file traversal helpers shared across retrieval modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from context_eng.ignore import path_is_ignored

# Extensions we treat as readable source/text for retrieval.
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".java",
    ".go", ".rs", ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".kt",
    ".swift", ".scala", ".sql", ".json", ".toml", ".yaml", ".yml", ".md",
    ".txt", ".cfg", ".ini",
}

_MAX_FILE_BYTES = 512 * 1024  # skip very large files

# Basename denylist (matched case-insensitively).
_SECRET_EXACT_NAMES = frozenset({
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "secrets.toml",
    "id_rsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
    "netrc",
    ".netrc",
})
_SECRET_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})
_SECRET_STEMS = frozenset({"credentials", "secrets"})
_SECRET_STEM_SUFFIXES = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".txt", ".ini", ".cfg",
})


def is_secret_path(path: Path) -> bool:
    """Return True if ``path`` looks like a secret-prone file (basename rules)."""
    name = path.name.lower()
    if name.startswith(".env"):
        return True
    if name in _SECRET_EXACT_NAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in _SECRET_SUFFIXES:
        return True
    if path.stem.lower() in _SECRET_STEMS and suffix in _SECRET_STEM_SUFFIXES:
        return True
    return False


def _is_ignored(rel_parts: tuple[str, ...], ignore_globs: tuple[str, ...]) -> bool:
    return path_is_ignored("/".join(rel_parts), ignore_globs)


def iter_files(
    workspace: Path, ignore_globs: tuple[str, ...]
) -> Iterator[Path]:
    """Yield text files under ``workspace`` skipping ignored directories."""
    workspace = workspace.resolve()
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if is_secret_path(path):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        rel = path.relative_to(workspace)
        if _is_ignored(rel.parts, ignore_globs):
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def read_text(path: Path) -> str:
    """Read a file as UTF-8 text, tolerating undecodable bytes."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def relpath(path: Path, workspace: Path) -> str:
    """POSIX-style path relative to the workspace root."""
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
