"""Tests for RF evaluation gates and engine budget integration."""

from pathlib import Path

from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.intent.classifier import analyze
from context_eng.ml.eval_rf import eval_cv, load_label_rows, write_report
from context_eng.ml.engine_budget import resolve_budget_limit

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks" / "fixture_repo"
LABELS = Path(__file__).resolve().parents[1] / "ml" / "data" / "budget_labels.jsonl"


def test_resolve_budget_limit_uses_intent_by_default():
    cfg = Config(workspace_root=FIXTURE, budget_source="intent")
    analysis = analyze("Why does refreshToken fail?", cfg)
    assert resolve_budget_limit("Why does refreshToken fail?", analysis, cfg, None) == (
        analysis.budget.recommended
    )


def test_resolve_budget_limit_honors_explicit_max_tokens():
    cfg = Config(workspace_root=FIXTURE, budget_source="rf")
    analysis = analyze("Why does refreshToken fail?", cfg)
    assert resolve_budget_limit("q", analysis, cfg, 5000) == 5000


def test_eval_cv_writes_pass_line(tmp_path):
    rows = load_label_rows(LABELS)
    cv = eval_cv(
        rows,
        {
            "min_cv_accuracy": 0.01,
            "min_cv_bucket_accuracy": 0.01,
            "min_cv_bucket_samples": 1000,
        },
    )
    anchor_line = "RF_ANCHOR_RECALL_GATE: PASS rf=1.000 intent=1.000 delta=0.000"
    ab_line = (
        "RF_AB_BENCHMARK_GATE: PASS rf_reduction=50.0 intent_reduction=50.0 "
        "rf_mcp=1000 intent_mcp=1000 regression_pp=0.0"
    )
    from context_eng.ml.eval_rf import AbBenchmarkEval, AnchorRecallEval

    write_report(
        tmp_path / "rf_eval.md",
        cv,
        AnchorRecallEval(1.0, 1.0, 0.0, True, anchor_line),
        AbBenchmarkEval(50.0, 50.0, 1000, 1000, 0.0, True, ab_line),
    )
    text = (tmp_path / "rf_eval.md").read_text(encoding="utf-8")
    assert "RF_CV_GATE: PASS" in text


def test_engine_accepts_rf_budget_source():
    cfg = Config(workspace_root=FIXTURE, budget_source="intent")
    engine = ContextEngine(config=cfg)
    bundle = engine.get_context_bundle("Why does refreshToken fail immediately after logout?")
    assert bundle.budget_limit > 0
