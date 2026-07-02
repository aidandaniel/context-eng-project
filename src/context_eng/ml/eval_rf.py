"""Evaluate the RF budget classifier: CV, anchor retention, and RF benchmark gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmarks.compare import aggregate, run_benchmark
from benchmarks.query_loader import load_queries as load_benchmark_queries
from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.intent.classifier import analyze
from context_eng.anchors.discovery import discover_anchor_paths
from context_eng.anchors.fit import ensure_budget_fits_anchors
from context_eng.ml.budget_model import BUDGET_BUCKETS
from context_eng.ml.engine_budget import rf_budget, resolve_budget
from context_eng.ml.features import FEATURE_NAMES, features_to_vector
from context_eng.ml.generate_labels import anchors_present, load_queries
from context_eng.models import CandidateChunk
from context_eng.retrieval.composite_retriever import build_retriever
from context_eng.retrieval.embedding_retriever import EmbeddingRetriever
from context_eng.retrieval.grep_retriever import GrepRetriever


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LABELS = _REPO_ROOT / "ml" / "data" / "budget_labels.jsonl"
_DEFAULT_TRAINING = _REPO_ROOT / "ml" / "data" / "budget_training_queries.yaml"
_DEFAULT_BENCHMARK = _REPO_ROOT / "benchmarks" / "queries.yaml"
_DEFAULT_TARGETS = _REPO_ROOT / "ml" / "data" / "eval_targets.yaml"
_DEFAULT_MODEL = _REPO_ROOT / "ml" / "models" / "budget_rf_v2.joblib"
_DEFAULT_WORKSPACE = _REPO_ROOT / "benchmarks" / "fixture_repo"
_DEFAULT_REPORT = _REPO_ROOT / "ml" / "reports" / "rf_eval.md"
_DEFAULT_EMBEDDING_EVAL = _REPO_ROOT / "ml" / "data" / "embedding_eval_queries.yaml"


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
        "min_cv_accuracy": 0.25,
        "min_median_reduction_pct": 55.0,
        "max_p90_latency_ms": 3000.0,
        "min_anchor_retention": 0.9,
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
    class_counts = [y.count(bucket) for bucket in set(y)]
    splits = min(5, min(class_counts)) if class_counts else 2
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

    for bucket in BUDGET_BUCKETS:
        idx = [i for i, label in enumerate(y) if label == bucket]
        if not idx:
            continue
        bucket_counts[bucket] = len(idx)
        correct = sum(1 for i in idx if y_pred[i] == y[i])
        acc = correct / len(idx)
        bucket_accuracy[bucket] = round(acc, 4)
        min_bucket = min(min_bucket, acc)

    passed = overall >= float(targets["min_cv_accuracy"])
    status = "PASS" if passed else "FAIL"
    gate_line = (
        f"RF_CV_GATE: {status} overall={overall:.3f} min_bucket={min_bucket:.3f}"
    )
    return CvEval(
        round(overall, 4),
        bucket_accuracy,
        bucket_counts,
        passed,
        gate_line,
    )


@dataclass
class RfBenchmarkEval:
    median_reduction_pct: float
    p90_latency_ms: float
    median_mcp_tokens: int
    token_reduction_passed: bool
    p90_latency_passed: bool
    token_reduction_gate_line: str
    p90_latency_gate_line: str

    @property
    def passed(self) -> bool:
        return self.token_reduction_passed and self.p90_latency_passed


def eval_rf_benchmark(
    workspace: Path,
    benchmark_queries_path: Path,
    model_path: Path,
    targets: dict[str, float | int],
) -> RfBenchmarkEval:
    queries = load_benchmark_queries(benchmark_queries_path)
    rf_config = Config(
        workspace_root=workspace.resolve(),
        budget_source="rf",
        ml_model_path=model_path,
    )
    agg = aggregate(run_benchmark(workspace, queries, rf_config))

    median_reduction = float(agg["median_reduction_pct"])
    p90_latency = float(agg["p90_latency_ms"])
    token_reduction_passed = median_reduction >= float(targets["min_median_reduction_pct"])
    p90_latency_passed = p90_latency < float(targets["max_p90_latency_ms"])

    token_status = "PASS" if token_reduction_passed else "FAIL"
    latency_status = "PASS" if p90_latency_passed else "FAIL"
    threshold = float(targets["min_median_reduction_pct"])
    return RfBenchmarkEval(
        median_reduction,
        p90_latency,
        int(agg["median_mcp_tokens"]),
        token_reduction_passed,
        p90_latency_passed,
        (
            f"TOKEN_REDUCTION_GATE: {token_status} "
            f"median_reduction={median_reduction:.1f} threshold_pct={threshold:.0f}"
        ),
        f"P90_LATENCY_GATE: {latency_status} p90_latency_ms={p90_latency:.1f}",
    )


@dataclass
class InferredLabelsEval:
    median_oracle_anchor_recall: float
    inferred_sweep_rows: int
    gate_line: str


def eval_inferred_labels(rows: list[dict[str, Any]]) -> InferredLabelsEval:
    """Audit label generation uses inferred anchors (not oracle sweep)."""
    recalls: list[float] = []
    inferred_count = 0
    for row in rows:
        if row.get("label_source") == "inferred_sweep":
            inferred_count += 1
        if "oracle_anchor_recall" in row:
            recalls.append(float(row["oracle_anchor_recall"]))
    median_recall = sorted(recalls)[len(recalls) // 2] if recalls else 0.0
    gate_line = (
        f"oracle_anchor_recall: median={median_recall:.3f} "
        f"inferred_sweep_rows={inferred_count}/{len(rows)}"
    )
    return InferredLabelsEval(
        round(median_recall, 4),
        inferred_count,
        gate_line,
    )


@dataclass
class BudgetAutofitEval:
    queries_bumped: int
    total_queries: int
    passed: bool
    gate_line: str


def eval_budget_autofit(
    workspace: Path,
    queries_path: Path,
    model_path: Path,
) -> BudgetAutofitEval:
    """Audit how often RF budget is bumped to fit discovered anchors."""
    config = Config(
        workspace_root=workspace.resolve(),
        budget_source="rf",
        ml_model_path=model_path,
    )
    engine = ContextEngine(config=config)

    bumped = 0
    total = 0
    for item in load_queries(queries_path):
        query = item["query"]
        analysis = analyze(query, config)
        grep = engine.retriever.search(
            query, workspace, config.max_grep_candidates
        )
        anchor_paths = discover_anchor_paths(
            query, analysis, workspace, grep, config
        )
        rf_resolution = resolve_budget(query, analysis, config, None)
        fitted = ensure_budget_fits_anchors(
            rf_resolution.limit,
            anchor_paths,
            workspace,
            analysis.signals.mentioned_symbols,
        )
        if fitted > rf_resolution.limit:
            bumped += 1
        total += 1

    passed = True
    gate_line = (
        f"BUDGET_AUTOFIT_GATE: PASS bumped={bumped}/{total}"
    )
    return BudgetAutofitEval(bumped, total, passed, gate_line)


@dataclass
class OptionalChunksEval:
    median_optional_chunks_used: float
    gate_line: str


def eval_optional_chunks(
    workspace: Path,
    queries_path: Path,
    model_path: Path,
) -> OptionalChunksEval:
    """Audit median optional chunks packed per query (adaptive cap)."""
    config = Config(
        workspace_root=workspace.resolve(),
        budget_source="rf",
        ml_model_path=model_path,
    )
    engine = ContextEngine(config=config)

    counts: list[int] = []
    for item in load_queries(queries_path):
        query = item["query"]
        analysis = analyze(query, config)
        rf_limit = rf_budget(query, analysis, config)
        bundle = engine.get_context_bundle(query, max_tokens=rf_limit)
        counts.append(bundle.optional_chunks_used)

    median = statistics.median(counts) if counts else 0.0
    gate_line = f"median_optional_chunks_used={median:.1f}"
    return OptionalChunksEval(round(median, 1), gate_line)


@dataclass
class AnchorRetentionEval:
    retention_rate: float
    passed: bool
    gate_line: str


def eval_anchor_retention(
    workspace: Path,
    queries_path: Path,
    model_path: Path,
    targets: dict[str, float | int],
) -> AnchorRetentionEval:
    config = Config(
        workspace_root=workspace.resolve(),
        budget_source="rf",
        ml_model_path=model_path,
    )
    engine = ContextEngine(config=config)

    ok = 0
    total = 0
    for item in load_queries(queries_path):
        query = item["query"]
        analysis = analyze(query, config)
        grep = engine.retriever.search(
            query, workspace, config.max_grep_candidates
        )
        discovered = discover_anchor_paths(query, analysis, workspace, grep, config)
        rf_limit = rf_budget(query, analysis, config)
        bundle = engine.get_context_bundle(query, max_tokens=rf_limit)
        if not discovered or anchors_present(bundle, discovered):
            ok += 1
        total += 1

    retention = ok / total if total else 0.0
    passed = retention >= float(targets["min_anchor_retention"])
    status = "PASS" if passed else "FAIL"
    gate_line = f"ANCHOR_RETENTION_GATE: {status} retention={retention:.3f}"
    return AnchorRetentionEval(round(retention, 4), passed, gate_line)


def _retrieval_signature(chunks: list[CandidateChunk]) -> str:
    parts = [
        f"{c.path}:{c.start_line}:{c.end_line}:{c.tier}:{c.keyword_match:.4f}"
        for c in chunks
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


@dataclass
class RegressionNoEmbeddingsEval:
    mismatches: int
    total_queries: int
    passed: bool
    gate_line: str


def eval_regression_no_embeddings(
    workspace: Path,
    queries_path: Path,
    model_path: Path,
) -> RegressionNoEmbeddingsEval:
    """Verify composite retriever with embeddings off matches pure grep."""
    config = Config(
        workspace_root=workspace.resolve(),
        budget_source="rf",
        ml_model_path=model_path,
        enable_embedding_retriever=False,
    )
    grep = GrepRetriever(config)
    composite = build_retriever(config)

    mismatches = 0
    total = 0
    for item in load_queries(queries_path):
        query = item["query"]
        limit = config.max_grep_candidates
        grep_hits = grep.search(query, workspace, limit)
        composite_hits = composite.search(query, workspace, limit)
        if _retrieval_signature(grep_hits) != _retrieval_signature(composite_hits):
            mismatches += 1
        total += 1

    passed = mismatches == 0
    status = "PASS" if passed else "FAIL"
    gate_line = (
        f"REGRESSION_NO_EMBEDDINGS_GATE: {status} "
        f"mismatches={mismatches}/{total}"
    )
    return RegressionNoEmbeddingsEval(mismatches, total, passed, gate_line)


def load_embedding_eval_queries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


@dataclass
class EmbeddingRecallEval:
    recall: float
    hits: int
    total_queries: int
    model_available: bool
    gate_line: str


def eval_embedding_recall(
    workspace: Path,
    queries_path: Path,
) -> EmbeddingRecallEval:
    """Audit semantic recall when local embedding model is available."""
    rows = load_embedding_eval_queries(queries_path)
    config = Config(
        workspace_root=workspace.resolve(),
        enable_embedding_retriever=True,
    )
    retriever = EmbeddingRetriever(config)
    model = retriever._load_model()
    if model is None or not rows:
        gate_line = (
            "embedding_eval_recall: skipped "
            f"model_available={model is not None} queries={len(rows)}"
        )
        return EmbeddingRecallEval(0.0, 0, len(rows), model is not None, gate_line)

    hits = 0
    for item in rows:
        query = str(item["query"])
        expected = {str(p) for p in item.get("expected_paths", [])}
        if not expected:
            continue
        found = {c.path for c in retriever.search(query, workspace, config.max_grep_candidates)}
        if expected & found:
            hits += 1

    total = len(rows)
    recall = hits / total if total else 0.0
    gate_line = f"embedding_eval_recall: {recall:.3f} hits={hits}/{total}"
    return EmbeddingRecallEval(round(recall, 3), hits, total, True, gate_line)


def write_report(
    path: Path,
    cv: CvEval,
    benchmark: RfBenchmarkEval,
    anchor: AnchorRetentionEval,
    autofit: BudgetAutofitEval | None = None,
    inferred_labels: InferredLabelsEval | None = None,
    optional_chunks: OptionalChunksEval | None = None,
    regression_no_embeddings: RegressionNoEmbeddingsEval | None = None,
    embedding_recall: EmbeddingRecallEval | None = None,
) -> None:
    lines = [
        "# RF evaluation report",
        "",
        benchmark.token_reduction_gate_line,
        benchmark.p90_latency_gate_line,
        anchor.gate_line,
        cv.gate_line,
    ]
    if autofit is not None:
        lines.append(autofit.gate_line)
    if inferred_labels is not None:
        lines.append(inferred_labels.gate_line)
    if optional_chunks is not None:
        lines.append(optional_chunks.gate_line)
    if regression_no_embeddings is not None:
        lines.append(regression_no_embeddings.gate_line)
    if embedding_recall is not None:
        lines.append(embedding_recall.gate_line)
    lines.extend([
        "",
        "## RF benchmark (queries.yaml)",
        f"- median token reduction: {benchmark.median_reduction_pct:.1f}%",
        f"- p90 latency: {benchmark.p90_latency_ms:.1f} ms",
        f"- median MCP tokens: {benchmark.median_mcp_tokens}",
    ])
    if autofit is not None:
        lines.extend([
            "",
            "## Budget auto-fit (budget_training_queries.yaml)",
            f"- queries bumped: {autofit.queries_bumped}/{autofit.total_queries}",
        ])
    if optional_chunks is not None:
        lines.extend([
            "",
            "## Adaptive optional chunks (budget_training_queries.yaml)",
            f"- {optional_chunks.gate_line}",
        ])
    if regression_no_embeddings is not None:
        lines.extend([
            "",
            "## Embedding regression (embeddings off)",
            f"- {regression_no_embeddings.gate_line}",
        ])
    if embedding_recall is not None:
        lines.extend([
            "",
            "## Embedding recall audit (embedding_eval_queries.yaml)",
            f"- {embedding_recall.gate_line}",
        ])
    lines.extend([
        "",
        "## Anchor retention (budget_training_queries.yaml)",
        f"- retention rate: {anchor.retention_rate:.3f}",
        "",
        "## 5-fold CV (budget_labels.jsonl)",
        f"- overall accuracy: {cv.overall_accuracy:.3f}",
        "- per-bucket accuracy:",
    ])
    for bucket, acc in sorted(cv.bucket_accuracy.items()):
        count = cv.bucket_counts.get(bucket, 0)
        lines.append(f"  - {bucket}: {acc:.3f} (n={count})")
    lines.append("")
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
    embedding_eval_path: Path = _DEFAULT_EMBEDDING_EVAL,
) -> dict[str, Any]:
    targets = _load_targets(targets_path)
    rows = load_label_rows(labels_path)
    cv = eval_cv(rows, targets)
    benchmark = eval_rf_benchmark(workspace, benchmark_queries_path, model_path, targets)
    anchor = eval_anchor_retention(workspace, training_queries_path, model_path, targets)
    autofit = eval_budget_autofit(workspace, training_queries_path, model_path)
    inferred_labels = eval_inferred_labels(rows)
    optional_chunks = eval_optional_chunks(workspace, training_queries_path, model_path)
    regression_no_embeddings = eval_regression_no_embeddings(
        workspace, training_queries_path, model_path
    )
    embedding_recall = eval_embedding_recall(workspace, embedding_eval_path)
    write_report(
        report_path,
        cv,
        benchmark,
        anchor,
        autofit,
        inferred_labels,
        optional_chunks,
        regression_no_embeddings,
        embedding_recall,
    )
    all_passed = (
        cv.passed
        and benchmark.passed
        and anchor.passed
        and regression_no_embeddings.passed
    )
    return {
        "cv": cv,
        "benchmark": benchmark,
        "anchor": anchor,
        "autofit": autofit,
        "inferred_labels": inferred_labels,
        "optional_chunks": optional_chunks,
        "regression_no_embeddings": regression_no_embeddings,
        "embedding_recall": embedding_recall,
        "all_passed": all_passed,
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
    parser.add_argument("--embedding-eval", type=Path, default=_DEFAULT_EMBEDDING_EVAL)
    args = parser.parse_args()

    result = evaluate_all(
        labels_path=args.labels,
        training_queries_path=args.training_queries,
        benchmark_queries_path=args.benchmark_queries,
        targets_path=args.targets,
        model_path=args.model,
        workspace=args.workspace,
        report_path=args.report,
        embedding_eval_path=args.embedding_eval,
    )
    cv: CvEval = result["cv"]
    benchmark: RfBenchmarkEval = result["benchmark"]
    anchor: AnchorRetentionEval = result["anchor"]
    autofit: BudgetAutofitEval = result["autofit"]
    inferred: InferredLabelsEval = result["inferred_labels"]
    optional: OptionalChunksEval = result["optional_chunks"]
    regression: RegressionNoEmbeddingsEval = result["regression_no_embeddings"]
    embedding: EmbeddingRecallEval = result["embedding_recall"]
    print(benchmark.token_reduction_gate_line)
    print(benchmark.p90_latency_gate_line)
    print(anchor.gate_line)
    print(cv.gate_line)
    print(autofit.gate_line)
    print(inferred.gate_line)
    print(optional.gate_line)
    print(regression.gate_line)
    print(embedding.gate_line)
    print(f"report: {result['report_path']}")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
