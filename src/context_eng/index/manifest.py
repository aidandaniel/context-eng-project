"""Cached workspace file manifest — avoids full rglob on every query."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_eng.config import Config
from context_eng.workspace import TEXT_EXTENSIONS, _MAX_FILE_BYTES, _is_ignored, relpath

_MANIFEST_VERSION = 1


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
            version=int(data.get("version", _MANIFEST_VERSION)),
            built_at=str(data.get("built_at", "")),
            entries=entries,
        )


def manifest_path(workspace: Path) -> Path:
    return workspace.resolve() / ".context-eng" / "manifest.json"


def _count_lines_quick(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    if not text:
        return 0
    return text.count("\n") + 1


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
    return WorkspaceManifest(
        workspace_root=workspace.as_posix(),
        version=_MANIFEST_VERSION,
        built_at=datetime.now(timezone.utc).isoformat(),
        entries=tuple(entries),
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
    return WorkspaceManifest.from_dict(data)


def _is_stale(manifest: WorkspaceManifest, workspace: Path) -> bool:
    if manifest.workspace_root != workspace.resolve().as_posix():
        return True
    ws = workspace.resolve()
    for entry in manifest.entries:
        path = ws / entry.rel_path
        if not path.is_file():
            return True
        try:
            stat = path.stat()
        except OSError:
            return True
        if stat.st_mtime_ns != entry.mtime_ns or stat.st_size != entry.size:
            return True
    return False


def get_manifest(workspace: Path, config: Config, *, rebuild: bool = False) -> WorkspaceManifest:
    """Return a fresh manifest, rebuilding when missing, stale, or forced."""
    workspace = workspace.resolve()
    if not rebuild and config.manifest_auto_build:
        cached = _load_manifest_file(workspace)
        if cached is not None and not _is_stale(cached, workspace):
            return cached
    manifest = build_manifest(workspace, config)
    if config.manifest_auto_build:
        save_manifest(manifest, workspace)
    return manifest


def get_searchable_files(workspace: Path, config: Config) -> list[Path]:
    """Absolute paths of searchable text files (manifest-backed)."""
    workspace = workspace.resolve()
    manifest = get_manifest(workspace, config)
    return [workspace / entry.rel_path for entry in manifest.entries]


def repo_stats_from_manifest(manifest: WorkspaceManifest) -> tuple[int, float]:
    """Return (file_count, log10(loc+1)) from manifest metadata."""
    loc = manifest.total_lines
    return manifest.file_count, math.log10(loc + 1)
