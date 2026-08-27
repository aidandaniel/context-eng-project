"""Tests for manifest-backed grep retrieval."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from unittest import mock

from context_eng.config import Config
from context_eng.retrieval import grep_retriever as gr
from context_eng.retrieval.grep_retriever import GrepRetriever, extract_keywords


def test_extract_keywords_splits_camel_case_and_dedupes():
    keywords = extract_keywords("Why does refreshToken fail after logout?")
    lowered = {k.lower() for k in keywords}
    assert "refresh" in lowered
    assert "token" in lowered
    assert "logout" in lowered
    assert "the" not in lowered


def test_search_finds_keyword_hits(tmp_path: Path):
    (tmp_path / "auth.py").write_text(
        textwrap.dedent(
            """
            def refreshToken(user):
                return user.token
            """
        ).strip(),
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("unrelated content\n", encoding="utf-8")
    config = Config(workspace_root=tmp_path, manifest_auto_build=True)
    hits = GrepRetriever(config).search("refreshToken failure", tmp_path, limit=5)
    assert hits
    assert hits[0].path == "auth.py"
    assert "refreshToken" in hits[0].content


def test_search_uses_manifest_not_iter_files(tmp_path: Path):
    (tmp_path / "target.py").write_text("needle_value = 42\n", encoding="utf-8")
    config = Config(workspace_root=tmp_path, manifest_auto_build=True)
    with mock.patch("context_eng.workspace.iter_files") as iter_files:
        hits = GrepRetriever(config).search("needle_value", tmp_path, limit=5)
    iter_files.assert_not_called()
    assert any(h.path == "target.py" for h in hits)


def test_ripgrep_path_when_available(tmp_path: Path):
    (tmp_path / "service.py").write_text("alpha_handler()\n", encoding="utf-8")
    config = Config(workspace_root=tmp_path, manifest_auto_build=True)
    fake_hits = {"service.py": [1]}
    with mock.patch.object(gr, "rg_available", return_value=True):
        with mock.patch.object(
            gr, "_ripgrep_file_hits", return_value=(fake_hits, True)
        ) as rg_hits:
            with mock.patch.object(gr, "_python_scan_hits") as py_hits:
                retriever = GrepRetriever(config)
                hits = retriever.search("alpha_handler", tmp_path, limit=5)
    rg_hits.assert_called_once()
    py_hits.assert_not_called()
    assert hits and hits[0].path == "service.py"
    assert retriever.last_degraded is False


def test_python_fallback_when_rg_missing(tmp_path: Path):
    (tmp_path / "beta.py").write_text("beta_handler = 1\n", encoding="utf-8")
    config = Config(workspace_root=tmp_path, manifest_auto_build=True)
    with mock.patch.object(gr, "rg_available", return_value=False):
        with mock.patch.object(gr, "_ripgrep_file_hits") as rg_hits:
            hits = GrepRetriever(config).search("beta_handler", tmp_path, limit=5)
    rg_hits.assert_not_called()
    assert any(h.path == "beta.py" for h in hits)


def test_rg_failure_falls_back_and_marks_degraded(tmp_path: Path):
    (tmp_path / "gamma.py").write_text("gamma_handler = 1\n", encoding="utf-8")
    config = Config(workspace_root=tmp_path, manifest_auto_build=True)
    with mock.patch.object(gr, "rg_available", return_value=True):
        with mock.patch.object(gr, "_ripgrep_file_hits", return_value=({}, False)):
            retriever = GrepRetriever(config)
            hits = retriever.search("gamma_handler", tmp_path, limit=5)
    assert any(h.path == "gamma.py" for h in hits)
    assert retriever.last_degraded is True


def test_chunks_from_hits_rejects_parent_escape(tmp_path: Path):
    (tmp_path / "ok.py").write_text("needle_value = 1\n", encoding="utf-8")
    outside = tmp_path.parent / "secret.py"
    # Do not require outside to exist; containment must fail first.
    patterns = [("needle_value", re.compile("needle_value"))]
    chunks = gr._chunks_from_hits(
        rel_path="../secret.py",
        hit_lines=[1],
        workspace=tmp_path,
        patterns=patterns,
        context=2,
    )
    assert chunks == []
    assert outside.name not in {c.path for c in chunks}


def test_parse_rg_json_drops_absolute_outside_path(tmp_path: Path):
    workspace = tmp_path
    payload = (
        '{"type":"match","data":{"path":{"text":"/etc/passwd"},'
        '"line_number":1}}\n'
    )
    hits = gr._parse_rg_json(payload, workspace)
    assert hits == {}
