"""Containment guards for poisoned manifest entries."""

from __future__ import annotations

import json
from pathlib import Path

from context_eng.config import Config
from context_eng.index.manifest import (
    contained_path,
    get_manifest,
    get_searchable_files,
    manifest_path,
)


def _config(workspace: Path) -> Config:
    return Config(workspace_root=workspace, manifest_auto_build=True)


def test_contained_path_rejects_parent_escape(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert contained_path(workspace, "../outside.txt") is None
    assert contained_path(workspace, "sub/../../outside.txt") is None


def test_contained_path_rejects_absolute(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = (tmp_path / "secret.txt").resolve()
    assert contained_path(workspace, str(outside)) is None


def test_contained_path_accepts_nested_relative(tmp_path: Path):
    workspace = tmp_path / "ws"
    nested = workspace / "pkg" / "mod.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("x = 1\n", encoding="utf-8")
    resolved = contained_path(workspace, "pkg/mod.py")
    assert resolved is not None
    assert resolved == nested.resolve()
    assert resolved.is_relative_to(workspace.resolve())


def test_poisoned_manifest_does_not_escape_workspace(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    inside = workspace / "a.py"
    inside.write_text("a = 1\n", encoding="utf-8")

    poisoned = {
        "version": 1,
        "workspace_root": workspace.resolve().as_posix(),
        "built_at": "2020-01-01T00:00:00+00:00",
        "entries": [
            {
                "rel_path": "../outside.txt",
                "mtime_ns": outside.stat().st_mtime_ns,
                "size": outside.stat().st_size,
                "line_count": 1,
            },
            {
                "rel_path": "a.py",
                "mtime_ns": inside.stat().st_mtime_ns,
                "size": inside.stat().st_size,
                "line_count": 1,
            },
        ],
    }
    path = manifest_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poisoned), encoding="utf-8")

    cfg = _config(workspace)
    paths = get_searchable_files(workspace, cfg)
    resolved = {p.resolve() for p in paths}

    assert outside.resolve() not in resolved
    assert all(p.is_relative_to(workspace.resolve()) for p in resolved)
    assert inside.resolve() in resolved


def test_poisoned_manifest_cache_is_discarded(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (workspace / "a.py").write_text("a = 1\n", encoding="utf-8")

    poisoned = {
        "version": 1,
        "workspace_root": workspace.resolve().as_posix(),
        "built_at": "2020-01-01T00:00:00+00:00",
        "entries": [
            {
                "rel_path": "../outside.txt",
                "mtime_ns": outside.stat().st_mtime_ns,
                "size": outside.stat().st_size,
                "line_count": 1,
            }
        ],
    }
    path = manifest_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poisoned), encoding="utf-8")

    cfg = _config(workspace)
    manifest = get_manifest(workspace, cfg, rebuild=False)
    assert all(not e.rel_path.startswith("..") for e in manifest.entries)
    assert {e.rel_path for e in manifest.entries} == {"a.py"}
