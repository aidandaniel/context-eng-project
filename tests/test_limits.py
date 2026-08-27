"""Query / budget / expand clamps."""

from __future__ import annotations

import pytest

from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.limits import MAX_BUDGET_TOKENS, MAX_QUERY_CHARS, validate_query
from context_eng.ml.budget_model import snap_to_bucket
from context_eng.ml.engine_budget import resolve_budget
from context_eng.models import BudgetInfo, Intent, QueryAnalysis, QuerySignals


def test_validate_query_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        validate_query("   ")


def test_validate_query_rejects_too_long():
    with pytest.raises(ValueError, match="exceeds"):
        validate_query("x" * (MAX_QUERY_CHARS + 1))


def test_snap_to_bucket_caps_explicit_budget():
    assert snap_to_bucket(1) == 2000
    assert snap_to_bucket(10_000_000) == MAX_BUDGET_TOKENS


def test_resolve_budget_snaps_explicit_override(tmp_path):
    cfg = Config(workspace_root=tmp_path)
    analysis = QueryAnalysis(
        intent=Intent.EXPLAIN,
        confidence=0.5,
        signals=QuerySignals(),
        budget=BudgetInfo(recommended=4000, min=2000, max=6000),
    )
    res = resolve_budget("explain greet", analysis, cfg, max_tokens=9_999_999)
    assert res.source == "explicit"
    assert res.limit == MAX_BUDGET_TOKENS


def test_expand_context_respects_max_expansions(tmp_path):
    (tmp_path / "hello.py").write_text("def greet():\n    return 1\n", encoding="utf-8")
    cfg = Config(workspace_root=tmp_path, max_expansions=1, manifest_auto_build=True)
    engine = ContextEngine(config=cfg)
    bundle = engine.get_context_bundle("explain greet in hello.py", max_tokens=2000)
    engine.expand_context(bundle.bundle_id)
    with pytest.raises(ValueError, match="limit reached"):
        engine.expand_context(bundle.bundle_id)


def test_engine_drops_escaped_anchor_paths(tmp_path, monkeypatch):
    (tmp_path / "ok.py").write_text("def greet():\n    return 1\n", encoding="utf-8")
    cfg = Config(workspace_root=tmp_path, manifest_auto_build=True)
    engine = ContextEngine(config=cfg)
    monkeypatch.setattr(
        "context_eng.engine.discover_anchor_paths",
        lambda *_a, **_k: ["../secret.py", "ok.py"],
    )
    bundle = engine.get_context_bundle("explain greet in ok.py", max_tokens=2000)
    paths = {c.path for c in bundle.chunks}
    assert all(not p.startswith("..") and "secret" not in p for p in paths)
