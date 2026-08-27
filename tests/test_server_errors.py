"""MCP server entrypoints must return error dicts instead of raising."""

from unittest.mock import MagicMock

import pytest

from context_eng.server import (
    _bundle_owners,
    _engines,
    analyze_query,
    context_prompt,
    expand_context,
    prepare_context,
)


@pytest.fixture(autouse=True)
def _clear_server_caches():
    _engines.clear()
    _bundle_owners.clear()
    yield
    _engines.clear()
    _bundle_owners.clear()


def _raise_permission_error(*_a, **_k):
    raise PermissionError("denied")


def test_prepare_context_permission_error_returns_dict(monkeypatch):
    monkeypatch.setattr("context_eng.server.get_engine", _raise_permission_error)
    result = prepare_context("explain greet")
    assert isinstance(result, dict)
    assert result["error"] == "denied"
    assert result["error_type"] == "PermissionError"


def test_analyze_query_permission_error_returns_dict(monkeypatch):
    monkeypatch.setattr("context_eng.server.get_engine", _raise_permission_error)
    result = analyze_query("explain greet")
    assert isinstance(result, dict)
    assert "error" in result
    assert result["error_type"] == "PermissionError"
    assert result["error"] == "denied"


def test_expand_context_engine_failure_returns_dict():
    engine = MagicMock()
    engine.expand_context.side_effect = PermissionError("denied")
    _bundle_owners["bid-1"] = engine

    result = expand_context("bid-1")
    assert isinstance(result, dict)
    assert result["error"] == "denied"
    assert result["error_type"] == "PermissionError"


def test_prepare_context_empty_query_returns_structured_error(tmp_path):
    result = prepare_context("   ", workspace_root=str(tmp_path))
    assert isinstance(result, dict)
    assert "error" in result
    assert result["error_type"] == "ValueError"


def test_write_tools_are_not_marked_read_only():
    import asyncio

    from context_eng.server import mcp

    tools = asyncio.run(mcp.list_tools())
    hints = {t.name: t.annotations.read_only_hint for t in tools}
    assert hints["prepare_context"] is False
    assert hints["get_context_bundle"] is False
    assert hints["analyze_query"] is False
    assert hints["expand_context"] is False
    assert hints["estimate_tokens"] is True
    assert hints["mcp_health"] is True
