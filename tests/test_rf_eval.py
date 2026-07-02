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


def test_resolve_budget_limit_uses_rf_by_default():
    from context_eng.ml.engine_budget import resolve_budget

    cfg = Config(workspace_root=FIXTURE)
    query = "Why does refreshToken fail?"
    analysis = analyze(query, cfg)
    resolution = resolve_budget(query, analysis, cfg, None)
    assert resolution.source == "rf"
    assert resolution.limit != analysis.budget.recommended or resolution.limit > 0


def test_resolve_budget_limit_falls_back_when_model_missing(tmp_path):
    from context_eng.ml.engine_budget import resolve_budget

    cfg = Config(
        workspace_root=FIXTURE,
        ml_model_path=tmp_path / "missing.joblib",
        default_max_tokens=8000,
    )
    analysis = analyze("query", cfg)
    resolution = resolve_budget("query", analysis, cfg, None)
    assert resolution.source == "fallback_default"
    assert resolution.limit == 8000


def test_resolve_budget_limit_honors_explicit_max_tokens():
    from context_eng.ml.engine_budget import resolve_budget

    cfg = Config(workspace_root=FIXTURE)
    analysis = analyze("Why does refreshToken fail?", cfg)
    resolution = resolve_budget("q", analysis, cfg, 5000)
    assert resolution.limit == 5000
    assert resolution.source == "explicit"


def test_eval_cv_writes_pass_line(tmp_path):
    rows = load_label_rows(LABELS)
    cv = eval_cv(rows, {"min_cv_accuracy": 0.01})
    benchmark = RfBenchmarkEval(
        50.0,
        100.0,
        1000,
        True,
        True,
        "TOKEN_REDUCTION_GATE: PASS median_reduction=50.0 threshold_pct=55.0",
        "P90_LATENCY_GATE: PASS p90_latency_ms=100.0",
    )
    anchor = AnchorRetentionEval(1.0, True, "ANCHOR_RETENTION_GATE: PASS retention=1.000")
    write_report(tmp_path / "rf_eval.md", cv, benchmark, anchor)
    text = (tmp_path / "rf_eval.md").read_text(encoding="utf-8")
    assert "RF_CV_GATE: PASS" in text
    assert "TOKEN_REDUCTION_GATE: PASS" in text
    assert "threshold_pct=55" in text
    assert "P90_LATENCY_GATE: PASS" in text
    assert "ANCHOR_RETENTION_GATE: PASS" in text


def test_engine_uses_rf_budget_by_default():
    cfg = Config(workspace_root=FIXTURE)
    engine = ContextEngine(config=cfg)
    bundle = engine.get_context_bundle("Why does refreshToken fail immediately after logout?")
    assert bundle.budget_limit > 0
