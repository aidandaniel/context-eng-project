"""Tests for workspace manifest cache."""

from __future__ import annotations

import json
import time
from pathlib import Path

from context_eng.config import Config
from context_eng.index.manifest import (
    build_manifest,
    get_manifest,
    get_searchable_files,
    manifest_path,
    save_manifest,
)


def _config(workspace: Path) -> Config:
    return Config(workspace_root=workspace, manifest_auto_build=True)


def test_build_manifest_lists_text_files(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text("def alpha():\n    pass\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("x", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01")

    manifest = build_manifest(tmp_path, _config(tmp_path))
    rel_paths = {entry.rel_path for entry in manifest.entries}
    assert "src/alpha.py" in rel_paths
    assert not any("node_modules" in p for p in rel_paths)
    assert not any(p.endswith(".bin") for p in rel_paths)
    assert manifest.total_lines >= 2


def test_save_and_load_manifest_roundtrip(tmp_path: Path):
    (tmp_path / "service.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    cfg = _config(tmp_path)
    built = build_manifest(tmp_path, cfg)
    path = save_manifest(built, tmp_path)
    assert path == manifest_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    loaded = get_manifest(tmp_path, cfg, rebuild=False)
    assert loaded.file_count == built.file_count
    assert loaded.workspace_root == built.workspace_root
    assert data["version"] == 1


def test_get_manifest_rebuilds_when_stale(tmp_path: Path):
    target = tmp_path / "module.py"
    target.write_text("v1\n", encoding="utf-8")
    cfg = _config(tmp_path)
    first = get_manifest(tmp_path, cfg)
    assert first.file_count == 1

    time.sleep(0.01)
    target.write_text("v1\nv2\n", encoding="utf-8")
    rebuilt = get_manifest(tmp_path, cfg)
    assert rebuilt.total_lines > first.total_lines


def test_get_searchable_files_returns_absolute_paths(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "handler.py").write_text("handler = 1\n", encoding="utf-8")
    cfg = _config(tmp_path)
    paths = get_searchable_files(tmp_path, cfg)
    assert len(paths) == 1
    assert paths[0].is_absolute()
    assert paths[0].name == "handler.py"
