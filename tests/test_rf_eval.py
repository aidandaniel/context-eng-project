"""Tests for RF evaluation gates and engine budget integration."""

from pathlib import Path

from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.intent.classifier import analyze
from context_eng.ml.eval_rf import (
    AnchorRetentionEval,
    CvEval,
    RfBenchmarkEval,
    eval_cv,
    load_label_rows,
    write_report,
)

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks" / "fixture_repo"
LABELS = Path(__file__).resolve().parents[1] / "ml" / "data" / "budget_labels.jsonl"


def test_resolve_budget_limit_uses_intent_by_default():
    from context_eng.ml.engine_budget import resolve_budget_limit

    cfg = Config(workspace_root=FIXTURE, budget_source="intent")
    analysis = analyze("Why does refreshToken fail?", cfg)
    assert resolve_budget_limit("Why does refreshToken fail?", analysis, cfg, None) == (
        analysis.budget.recommended
    )


def test_resolve_budget_limit_honors_explicit_max_tokens():
    from context_eng.ml.engine_budget import resolve_budget_limit

    cfg = Config(workspace_root=FIXTURE, budget_source="rf")
    analysis = analyze("Why does refreshToken fail?", cfg)
    assert resolve_budget_limit("q", analysis, cfg, 5000) == 5000


def test_eval_cv_writes_pass_line(tmp_path):
    rows = load_label_rows(LABELS)
    cv = eval_cv(rows, {"min_cv_accuracy": 0.01})
    benchmark = RfBenchmarkEval(
        50.0,
        100.0,
        1000,
        True,
        True,
        "TOKEN_REDUCTION_GATE: PASS median_reduction=50.0",
        "P90_LATENCY_GATE: PASS p90_latency_ms=100.0",
    )
    anchor = AnchorRetentionEval(1.0, True, "ANCHOR_RETENTION_GATE: PASS retention=1.000")
    write_report(tmp_path / "rf_eval.md", cv, benchmark, anchor)
    text = (tmp_path / "rf_eval.md").read_text(encoding="utf-8")
    assert "RF_CV_GATE: PASS" in text
    assert "TOKEN_REDUCTION_GATE: PASS" in text
    assert "P90_LATENCY_GATE: PASS" in text
    assert "ANCHOR_RETENTION_GATE: PASS" in text


def test_engine_accepts_rf_budget_source():
    cfg = Config(workspace_root=FIXTURE, budget_source="intent")
    engine = ContextEngine(config=cfg)
    bundle = engine.get_context_bundle("Why does refreshToken fail immediately after logout?")
    assert bundle.budget_limit > 0
