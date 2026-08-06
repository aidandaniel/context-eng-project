"""Tests for workspace path resolution allowlist jail."""

from __future__ import annotations

from pathlib import Path

import pytest

from context_eng.workspace_resolve import resolve_workspace


def test_allows_cwd_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONTEXT_ENG_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("CONTEXT_ENG_WORKSPACE", raising=False)
    child = tmp_path / "proj"
    child.mkdir()
    assert resolve_workspace(str(child)) == child.resolve()


def test_rejects_sibling_outside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    sibling = tmp_path / "sibling"
    cwd.mkdir()
    sibling.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("CONTEXT_ENG_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("CONTEXT_ENG_WORKSPACE", raising=False)
    with pytest.raises(PermissionError, match="CONTEXT_ENG_ALLOWED_ROOTS"):
        resolve_workspace(str(sibling))


def test_allowed_roots_permits_extra_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    extra = tmp_path / "extra"
    cwd.mkdir()
    extra.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("CONTEXT_ENG_WORKSPACE", raising=False)
    monkeypatch.setenv("CONTEXT_ENG_ALLOWED_ROOTS", str(extra.resolve()))
    assert resolve_workspace(str(extra)) == extra.resolve()


def test_unset_env_defaults_to_cwd_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    outside = tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("CONTEXT_ENG_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("CONTEXT_ENG_WORKSPACE", raising=False)
    assert resolve_workspace(None) == cwd.resolve()
    with pytest.raises(PermissionError, match="allowlist"):
        resolve_workspace(str(outside))
