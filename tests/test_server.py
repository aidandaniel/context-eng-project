"""Tests for multi-project workspace resolution and MCP server routing."""

from pathlib import Path

import pytest

from context_eng.server import (
    _bundle_owners,
    _engines,
    context_prompt,
    expand_context,
    get_context_bundle,
    get_engine,
    prepare_context,
)
from context_eng.workspace_resolve import resolve_workspace


@pytest.fixture(autouse=True)
def _clear_server_caches():
    """Isolate tests from cached engines/bundles."""
    _engines.clear()
    _bundle_owners.clear()
    yield
    _engines.clear()
    _bundle_owners.clear()


def test_resolve_workspace_explicit(tmp_path):
    sub = tmp_path / "proj"
    sub.mkdir()
    assert resolve_workspace(str(sub)) == sub.resolve()


def test_resolve_workspace_prefers_explicit_over_env(tmp_path, monkeypatch):
    env_dir = tmp_path / "from_env"
    explicit = tmp_path / "from_arg"
    env_dir.mkdir()
    explicit.mkdir()
    monkeypatch.setenv("CONTEXT_ENG_WORKSPACE", str(env_dir))
    assert resolve_workspace(str(explicit)) == explicit.resolve()


def test_get_engine_caches_per_workspace(tmp_path):
    a = tmp_path / "repo_a"
    b = tmp_path / "repo_b"
    a.mkdir()
    b.mkdir()
    ea = get_engine(str(a))
    eb = get_engine(str(b))
    assert get_engine(str(a)) is ea
    assert get_engine(str(b)) is eb
    assert ea is not eb
    assert ea.config.workspace_root == a.resolve()
    assert eb.config.workspace_root == b.resolve()


def test_get_context_bundle_tracks_workspace_for_expand(tmp_path):
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / "hello.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")

    result = get_context_bundle("explain greet in hello.py", workspace_root=str(repo))
    assert "error" not in result
    assert Path(result["workspace_root"]) == repo.resolve()
    bundle_id = result["bundle_id"]

    expanded = expand_context(bundle_id)
    assert "error" not in expanded
    assert expanded["bundle_id"] == bundle_id


def test_expand_context_unknown_bundle():
    result = expand_context("nonexistent-bundle-id")
    assert "error" in result


def test_prepare_context_one_shot(tmp_path):
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / "hello.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")

    result = prepare_context("explain greet in hello.py", workspace_root=str(repo))
    assert "error" not in result
    assert result["query"] == "explain greet in hello.py"
    assert "analysis" in result
    assert "bundle" in result
    assert "formatted_context" in result
    assert "hello.py" in result["formatted_context"]
    assert result["bundle"]["bundle_id"]


def test_context_prompt_returns_formatted_context(tmp_path):
    repo = tmp_path / "fixture"
    repo.mkdir()
    (repo / "hello.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")

    text = context_prompt("explain greet in hello.py", workspace_root=str(repo))
    assert "Context Engineering" in text
    assert "hello.py" in text
    assert "bundle id" in text.lower()
