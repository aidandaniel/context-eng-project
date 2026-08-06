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


def test_context_prompt_permission_error_returns_string(monkeypatch):
    monkeypatch.setattr("context_eng.server.get_engine", _raise_permission_error)
    text = context_prompt("explain greet")
    assert isinstance(text, str)
    assert "denied" in text
    assert "Traceback" not in text
