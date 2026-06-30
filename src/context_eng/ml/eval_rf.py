"""Evaluate the RF budget classifier: CV, anchor recall, and A/B token benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from benchmarks.compare import aggregate, run_benchmark
from benchmarks.query_loader import load_queries as load_benchmark_queries
from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.intent.classifier import analyze
from context_eng.ml.budget_model import BUDGET_BUCKETS
from context_eng.ml.engine_budget import rf_budget
from context_eng.ml.features import FEATURE_NAMES, features_to_vector
from context_eng.ml.generate_labels import anchors_present, load_queries

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LABELS = _REPO_ROOT / "ml" / "data" / "budget_labels.jsonl"
_DEFAULT_TRAINING = _REPO_ROOT / "ml" / "data" / "budget_training_queries.yaml"
_DEFAULT_BENCHMARK = _REPO_ROOT / "benchmarks" / "queries.yaml"
_DEFAULT_TARGETS = _REPO_ROOT / "ml" / "data" / "eval_targets.yaml"
_DEFAULT_MODEL = _REPO_ROOT / "ml" / "models" / "budget_rf_v2.joblib"
_DEFAULT_WORKSPACE = _REPO_ROOT / "benchmarks" / "fixture_repo"
_DEFAULT_REPORT = _REPO_ROOT / "ml" / "reports" / "rf_eval.md"
_TRAINING_OVERRIDES = {
    "max_grep_candidates": 200,
    "max_inferred_anchor_files": 40,
    "inferred_anchor_min_score": 0.0,
    "max_optional_chunks": 16,
    "min_chunk_score": 0.0,
}


def _require_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError as exc:
        raise RuntimeError(
            "RF evaluation requires the 'ml' extra: pip install -e '.[ml]'"
        ) from exc
    return RandomForestClassifier, StratifiedKFold, cross_val_predict


def _load_targets(path: Path) -> dict[str, float | int]:
    defaults: dict[str, float | int] = {
        "min_cv_accuracy": 0.28,
        "min_cv_bucket_accuracy": 0.15,
        "min_cv_bucket_samples": 10,
        "min_anchor_recall_rf": 0.72,
        "min_anchor_recall_delta": 0.0,
        "max_reduction_regression_pp": 5.0,
    }
    if not path.is_file():
        return defaults
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in defaults:
            defaults[key] = float(value) if "." in value else int(value)
    return defaults


def load_label_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass
class CvEval:
    overall_accuracy: float
    bucket_accuracy: dict[int, float]
    bucket_counts: dict[int, int]
    passed: bool
    gate_line: str


def eval_cv(rows: list[dict[str, Any]], targets: dict[str, float | int]) -> CvEval:
    RandomForestClassifier, StratifiedKFold, cross_val_predict = _require_sklearn()
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        values, names = features_to_vector(row["features"])
        if names != FEATURE_NAMES:
            raise ValueError("label row feature order does not match FEATURE_NAMES")
        x.append(values)
        y.append(int(row["y"]))

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=17,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
    )
    splits = min(5, min(y.count(b) for b in set(y)))
    if splits < 2:
        return CvEval(
            0.0,
            {},
            {},
            False,
            "RF_CV_GATE: FAIL reason=insufficient_class_samples",
        )

    y_pred = cross_val_predict(
        classifier,
        x,
        y,
        cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=17),
    )

    overall = sum(int(a == b) for a, b in zip(y, y_pred, strict=True)) / len(y)
    bucket_accuracy: dict[int, float] = {}
    bucket_counts: dict[int, int] = {}
    min_bucket = 1.0
    min_samples = int(targets["min_cv_bucket_samples"])
    bucket_failures: list[str] = []

    for bucket in BUDGET_BUCKETS:
        idx = [i for i, label in enumerate(y) if label == bucket]
        if not idx:
            continue
        bucket_counts[bucket] = len(idx)
        if len(idx) < min_samples:
            continue
        correct = sum(1 for i in idx if y_pred[i] == y[i])
        acc = correct / len(idx)
        bucket_accuracy[bucket] = round(acc, 4)
        min_bucket = min(min_bucket, acc)
        if acc < float(targets["min_cv_bucket_accuracy"]):
            bucket_failures.append(f"{bucket}={acc:.3f}")

    passed = overall >= float(targets["min_cv_accuracy"]) and not bucket_failures
    status = "PASS" if passed else "FAIL"
    gate_line = (
        f"RF_CV_GATE: {status} overall={overall:.3f} min_bucket={min_bucket:.3f}"
    )
    if bucket_failures:
        gate_line += f" weak_buckets={','.join(bucket_failures)}"
    return CvEval(
        round(overall, 4),
        bucket_accuracy,
        bucket_counts,
        passed,
        gate_line,
    )


@dataclass
class AnchorRecallEval:
    rf_recall: float
    intent_recall: float
    delta: float
    passed: bool
    gate_line: str


def eval_anchor_recall(
    workspace: Path,
    queries_path: Path,
    targets: dict[str, float | int],
) -> AnchorRecallEval:
    config = Config(workspace_root=workspace.resolve(), budget_source="rf")
    config = replace(config, **_TRAINING_OVERRIDES)
    engine = ContextEngine(config=config)

    rf_ok = 0
    intent_ok = 0
    total = 0
    for item in load_queries(queries_path):
        query = item["query"]
        anchors = item.get("expected_anchors", [])
        analysis = analyze(query, config)
        rf_limit = rf_budget(query, analysis, config)
        intent_limit = analysis.budget.recommended

        rf_bundle = engine.get_context_bundle(query, max_tokens=rf_limit)
        intent_bundle = engine.get_context_bundle(query, max_tokens=intent_limit)

        if anchors_present(rf_bundle, anchors):
            rf_ok += 1
        if anchors_present(intent_bundle, anchors):
            intent_ok += 1
        total += 1

    rf_recall = rf_ok / total if total else 0.0
    intent_recall = intent_ok / total if total else 0.0
    delta = rf_recall - intent_recall
    passed = (
        rf_recall >= float(targets["min_anchor_recall_rf"])
        and delta >= float(targets["min_anchor_recall_delta"])
    )
    status = "PASS" if passed else "FAIL"
    gate_line = (
        f"RF_ANCHOR_RECALL_GATE: {status} rf={rf_recall:.3f} "
        f"intent={intent_recall:.3f} delta={delta:.3f}"
    )
    return AnchorRecallEval(
        round(rf_recall, 4),
        round(intent_recall, 4),
        round(delta, 4),
        passed,
        gate_line,
    )


@dataclass
class AbBenchmarkEval:
    intent_reduction: float
    rf_reduction: float
    intent_mcp_tokens: int
    rf_mcp_tokens: int
    regression_pp: float
    passed: bool
    gate_line: str


def eval_ab_benchmark(
    workspace: Path,
    benchmark_queries_path: Path,
    model_path: Path,
    targets: dict[str, float | int],
) -> AbBenchmarkEval:
    queries = load_benchmark_queries(benchmark_queries_path)

    intent_config = Config(workspace_root=workspace.resolve(), budget_source="intent")
    rf_config = Config(
        workspace_root=workspace.resolve(),
        budget_source="rf",
        ml_model_path=model_path,
    )
    intent_agg = aggregate(run_benchmark(workspace, queries, intent_config))
    rf_agg = aggregate(run_benchmark(workspace, queries, rf_config))

    regression = intent_agg["median_reduction_pct"] - rf_agg["median_reduction_pct"]
    passed = regression <= float(targets["max_reduction_regression_pp"])
    status = "PASS" if passed else "FAIL"
    gate_line = (
        f"RF_AB_BENCHMARK_GATE: {status} rf_reduction={rf_agg['median_reduction_pct']:.1f} "
        f"intent_reduction={intent_agg['median_reduction_pct']:.1f} "
        f"rf_mcp={rf_agg['median_mcp_tokens']} intent_mcp={intent_agg['median_mcp_tokens']} "
        f"regression_pp={regression:.1f}"
    )
    return AbBenchmarkEval(
        intent_agg["median_reduction_pct"],
        rf_agg["median_reduction_pct"],
        rf_agg["median_mcp_tokens"],
        intent_agg["median_mcp_tokens"],
        round(regression, 2),
        passed,
        gate_line,
    )


def write_report(
    path: Path,
    cv: CvEval,
    anchor: AnchorRecallEval,
    ab: AbBenchmarkEval,
) -> None:
    lines = [
        "# RF evaluation report",
        "",
        cv.gate_line,
        anchor.gate_line,
        ab.gate_line,
        "",
        "## 5-fold CV (budget_labels.jsonl)",
        f"- overall accuracy: {cv.overall_accuracy:.3f}",
        "- per-bucket accuracy (buckets with enough samples):",
    ]
    for bucket, acc in sorted(cv.bucket_accuracy.items()):
        count = cv.bucket_counts.get(bucket, 0)
        lines.append(f"  - {bucket}: {acc:.3f} (n={count})")
    lines.extend(
        [
            "",
            "## Anchor recall (budget_training_queries.yaml)",
            f"- RF recall: {anchor.rf_recall:.3f}",
            f"- Intent recall: {anchor.intent_recall:.3f}",
            f"- Delta (RF - intent): {anchor.delta:.3f}",
            "",
            "## A/B token benchmark (benchmarks/queries.yaml)",
            f"- Intent median reduction: {ab.intent_reduction:.1f}%",
            f"- RF median reduction: {ab.rf_reduction:.1f}%",
            f"- Regression (intent - RF): {ab.regression_pp:.1f} pp",
            f"- Intent median MCP tokens: {ab.intent_mcp_tokens}",
            f"- RF median MCP tokens: {ab.rf_mcp_tokens}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_all(
    *,
    labels_path: Path = _DEFAULT_LABELS,
    training_queries_path: Path = _DEFAULT_TRAINING,
    benchmark_queries_path: Path = _DEFAULT_BENCHMARK,
    targets_path: Path = _DEFAULT_TARGETS,
    model_path: Path = _DEFAULT_MODEL,
    workspace: Path = _DEFAULT_WORKSPACE,
    report_path: Path = _DEFAULT_REPORT,
) -> dict[str, Any]:
    targets = _load_targets(targets_path)
    rows = load_label_rows(labels_path)
    cv = eval_cv(rows, targets)
    anchor = eval_anchor_recall(workspace, training_queries_path, targets)
    ab = eval_ab_benchmark(workspace, benchmark_queries_path, model_path, targets)
    write_report(report_path, cv, anchor, ab)
    return {
        "cv": cv,
        "anchor": anchor,
        "ab": ab,
        "all_passed": cv.passed and anchor.passed and ab.passed,
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RF budget classifier gates.")
    parser.add_argument("--labels", type=Path, default=_DEFAULT_LABELS)
    parser.add_argument("--training-queries", type=Path, default=_DEFAULT_TRAINING)
    parser.add_argument("--benchmark-queries", type=Path, default=_DEFAULT_BENCHMARK)
    parser.add_argument("--targets", type=Path, default=_DEFAULT_TARGETS)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--workspace", type=Path, default=_DEFAULT_WORKSPACE)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    args = parser.parse_args()

    result = evaluate_all(
        labels_path=args.labels,
        training_queries_path=args.training_queries,
        benchmark_queries_path=args.benchmark_queries,
        targets_path=args.targets,
        model_path=args.model,
        workspace=args.workspace,
        report_path=args.report,
    )
    cv: CvEval = result["cv"]
    anchor: AnchorRecallEval = result["anchor"]
    ab: AbBenchmarkEval = result["ab"]
    print(cv.gate_line)
    print(anchor.gate_line)
    print(ab.gate_line)
    print(f"report: {result['report_path']}")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
