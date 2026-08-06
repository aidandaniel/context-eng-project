"""H2: repo_stats uses manifest metadata, not full-file reads."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from context_eng.config import Config
from context_eng.ml.repo_stats import clear_repo_stats_cache, repo_stats


def test_repo_stats_uses_manifest_not_read_text(tmp_path: Path):
    (tmp_path / "a.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("only\n", encoding="utf-8")
    cfg = Config(workspace_root=tmp_path, manifest_auto_build=True)
    clear_repo_stats_cache()

    with mock.patch("context_eng.ml.repo_stats.read_text") as read_text:
        count, loc_log = repo_stats(cfg)
        read_text.assert_not_called()

    assert count == 2
    assert loc_log > 0


def test_repo_stats_process_cache_skips_rebuild(tmp_path: Path):
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    cfg = Config(workspace_root=tmp_path, manifest_auto_build=True)
    clear_repo_stats_cache()
    first = repo_stats(cfg)
    with mock.patch("context_eng.index.manifest.get_manifest") as get_manifest:
        second = repo_stats(cfg)
        get_manifest.assert_not_called()
    assert second == first


def test_repo_stats_walk_fallback_when_manifest_disabled(tmp_path: Path):
    (tmp_path / "x.py").write_text("a\nb\n", encoding="utf-8")
    cfg = Config(workspace_root=tmp_path, manifest_auto_build=False)
    clear_repo_stats_cache()
    count, loc_log = repo_stats(cfg)
    assert count == 1
    assert loc_log > 0
