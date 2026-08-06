"""Cached workspace file manifest — avoids full rglob on every query."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_eng.config import Config
from context_eng.workspace import (
    TEXT_EXTENSIONS,
    _MAX_FILE_BYTES,
    _is_ignored,
    is_secret_path,
)

# v2 adds dir watermark + path-set hash so newly added files invalidate cache.
_MANIFEST_VERSION = 2


@dataclass(frozen=True)
class ManifestEntry:
    rel_path: str
    mtime_ns: int
    size: int
    line_count: int


@dataclass(frozen=True)
class WorkspaceManifest:
    workspace_root: str
    version: int
    built_at: str
    entries: tuple[ManifestEntry, ...]
    dir_watermark_ns: int = 0
    path_set_hash: str = ""
    entry_count: int = 0

    @property
    def file_count(self) -> int:
        return len(self.entries)

    @property
    def total_lines(self) -> int:
        return sum(e.line_count for e in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "workspace_root": self.workspace_root,
            "built_at": self.built_at,
            "dir_watermark_ns": self.dir_watermark_ns,
            "path_set_hash": self.path_set_hash,
            "entry_count": self.entry_count or len(self.entries),
            "entries": [
                {
                    "rel_path": e.rel_path,
                    "mtime_ns": e.mtime_ns,
                    "size": e.size,
                    "line_count": e.line_count,
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkspaceManifest:
        entries = tuple(
            ManifestEntry(
                rel_path=str(item["rel_path"]),
                mtime_ns=int(item["mtime_ns"]),
                size=int(item["size"]),
                line_count=int(item.get("line_count", 0)),
            )
            for item in data.get("entries", [])
        )
        return cls(
            workspace_root=str(data["workspace_root"]),
            version=int(data.get("version", 1)),
            built_at=str(data.get("built_at", "")),
            entries=entries,
            dir_watermark_ns=int(data.get("dir_watermark_ns", 0)),
            path_set_hash=str(data.get("path_set_hash", "")),
            entry_count=int(data.get("entry_count", len(entries))),
        )


def manifest_path(workspace: Path) -> Path:
    return workspace.resolve() / ".context-eng" / "manifest.json"


def contained_path(workspace: Path, rel_path: str) -> Path | None:
    """Resolve ``rel_path`` under ``workspace`` only if it stays contained.

    Rejects absolute paths and ``..`` (or symlink) escapes. Returns the
    resolved path on success, otherwise ``None``.
    """
    if not rel_path:
        return None
    candidate = Path(rel_path)
    if candidate.is_absolute():
        return None
    ws = workspace.resolve()
    try:
        resolved = (ws / candidate).resolve()
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(ws):
        return None
    return resolved


def _count_lines_quick(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    if not text:
        return 0
    return text.count("\n") + 1


def _path_set_hash(paths: set[str] | list[str]) -> str:
    blob = "\n".join(sorted(paths)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _dir_watermark_ns(workspace: Path, config: Config) -> int:
    """Max directory mtime under workspace (respecting ignore_globs).

    Creating/renaming a file typically updates its parent directory mtime, so
    this detects path-set growth without reading file contents.
    """
    workspace = workspace.resolve()
    watermark = 0
    try:
        watermark = workspace.stat().st_mtime_ns
    except OSError:
        pass
    for path in workspace.rglob("*"):
        if not path.is_dir():
            continue
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue
        if _is_ignored(rel.parts, config.ignore_globs):
            continue
        try:
            watermark = max(watermark, path.stat().st_mtime_ns)
        except OSError:
            continue
    return watermark


def _searchable_rel_paths(workspace: Path, config: Config) -> set[str]:
    """Lightweight rescan of searchable relative paths (no line counting)."""
    workspace = workspace.resolve()
    found: set[str] = set()
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if is_secret_path(path):
            continue
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue
        if _is_ignored(rel.parts, config.ignore_globs):
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        found.add(rel.as_posix())
    return found


def build_manifest(workspace: Path, config: Config) -> WorkspaceManifest:
    """Walk workspace once and build a searchable file manifest."""
    from datetime import datetime, timezone

    workspace = workspace.resolve()
    entries: list[ManifestEntry] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if is_secret_path(path):
            continue
        rel = path.relative_to(workspace)
        if _is_ignored(rel.parts, config.ignore_globs):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size > _MAX_FILE_BYTES:
            continue
        entries.append(
            ManifestEntry(
                rel_path=rel.as_posix(),
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
                line_count=_count_lines_quick(path),
            )
        )
    entries.sort(key=lambda e: e.rel_path)
    paths = {e.rel_path for e in entries}
    return WorkspaceManifest(
        workspace_root=workspace.as_posix(),
        version=_MANIFEST_VERSION,
        built_at=datetime.now(timezone.utc).isoformat(),
        entries=tuple(entries),
        dir_watermark_ns=_dir_watermark_ns(workspace, config),
        path_set_hash=_path_set_hash(paths),
        entry_count=len(entries),
    )


def save_manifest(manifest: WorkspaceManifest, workspace: Path) -> Path:
    path = manifest_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def _load_manifest_file(workspace: Path) -> WorkspaceManifest | None:
    path = manifest_path(workspace)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    manifest = WorkspaceManifest.from_dict(data)
    ws = workspace.resolve()
    # Fail closed: discard cache if any entry escapes the workspace.
    if any(contained_path(ws, entry.rel_path) is None for entry in manifest.entries):
        return None
    return manifest


def _entries_stale(manifest: WorkspaceManifest, workspace: Path) -> bool:
    ws = workspace.resolve()
    for entry in manifest.entries:
        path = contained_path(ws, entry.rel_path)
        if path is None or not path.is_file():
            return True
        try:
            stat = path.stat()
        except OSError:
            return True
        if stat.st_mtime_ns != entry.mtime_ns or stat.st_size != entry.size:
            return True
    return False


def _is_stale(manifest: WorkspaceManifest, workspace: Path, config: Config) -> bool:
    if manifest.version != _MANIFEST_VERSION:
        return True
    if manifest.workspace_root != workspace.resolve().as_posix():
        return True
    if _entries_stale(manifest, workspace):
        return True

    # Fast path: unchanged directory watermark + entry count ⇒ no path-set growth.
    live_wm = _dir_watermark_ns(workspace, config)
    expected_count = manifest.entry_count or len(manifest.entries)
    if (
        live_wm == manifest.dir_watermark_ns
        and expected_count == len(manifest.entries)
        and manifest.path_set_hash
    ):
        return False

    # Directory churn or missing hash: compare searchable path set.
    disk_paths = _searchable_rel_paths(workspace.resolve(), config)
    if len(disk_paths) != expected_count:
        return True
    if manifest.path_set_hash:
        return _path_set_hash(disk_paths) != manifest.path_set_hash
    manifest_paths = {entry.rel_path for entry in manifest.entries}
    return disk_paths != manifest_paths


def get_manifest(workspace: Path, config: Config, *, rebuild: bool = False) -> WorkspaceManifest:
    """Return a fresh manifest, rebuilding when missing, stale, or forced."""
    workspace = workspace.resolve()
    if not rebuild and config.manifest_auto_build:
        cached = _load_manifest_file(workspace)
        if cached is not None and not _is_stale(cached, workspace, config):
            return cached
    manifest = build_manifest(workspace, config)
    if config.manifest_auto_build:
        save_manifest(manifest, workspace)
    return manifest


def get_searchable_files(workspace: Path, config: Config) -> list[Path]:
    """Absolute paths of searchable text files (manifest-backed)."""
    workspace = workspace.resolve()
    manifest = get_manifest(workspace, config)
    paths: list[Path] = []
    for entry in manifest.entries:
        path = contained_path(workspace, entry.rel_path)
        if path is not None:
            paths.append(path)
    return paths


def repo_stats_from_manifest(manifest: WorkspaceManifest) -> tuple[int, float]:
    """Return (file_count, log10(loc+1)) from manifest metadata."""
    loc = manifest.total_lines
    return manifest.file_count, math.log10(loc + 1)
