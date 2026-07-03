"""Generate budget-bucket labels from the RF training corpus."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from context_eng.anchors.discovery import discover_anchor_paths
from context_eng.anchors.fit import estimate_must_include_tokens
from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.eval.quality import relevant_file_recall
from context_eng.intent.classifier import analyze
from context_eng.ml.budget_model import BUDGET_BUCKETS, snap_to_bucket, snap_to_bucket
from context_eng.ml.features import extract_features
from benchmarks.query_loader import load_queries as load_benchmark_queries

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WORKSPACE = _REPO_ROOT / "benchmarks" / "fixture_repo"
_DEFAULT_QUERIES = _REPO_ROOT / "ml" / "data" / "budget_training_queries.yaml"
_DEFAULT_OUTPUT = _REPO_ROOT / "ml" / "data" / "budget_labels.jsonl"


def anchors_present(bundle: Any, anchor_paths: list[str]) -> bool:
    """Return whether every anchor path is represented in ``bundle``."""
    paths = {c.path for c in bundle.chunks}
    for anchor in anchor_paths:
        normalized = anchor.replace("\\", "/")
        if not any(p == normalized or p.endswith(normalized) for p in paths):
            return False
    return True


def oracle_anchor_recall(
    expected_anchors: list[str],
    discovered_anchors: list[str],
) -> float:
    """Audit recall of oracle labels against runtime-discovered anchors."""
    if not expected_anchors:
        return 1.0
    discovered = {p.replace("\\", "/") for p in discovered_anchors}
    hits = 0
    for anchor in expected_anchors:
        normalized = anchor.replace("\\", "/")
        if any(p == normalized or p.endswith(normalized) for p in discovered):
            hits += 1
    return hits / len(expected_anchors)


def adaptive_inferred_anchor_limit(analysis: Any, config: Config) -> int:
    """Scale inferred-anchor cap with query breadth (runtime-shaped, no oracle)."""
    base = config.max_inferred_anchor_files
    boost = len(analysis.signals.mentioned_symbols) // 4
    return min(config.max_inferred_anchor_files_upper, base + boost)


def sweep_one_query_quality(
    query: str,
    config: Config,
    expected_anchors: list[str],
) -> dict[str, Any]:
    """Find smallest bucket with recall-aware fit and token ceiling."""
    analysis = analyze(query, config)
    anchor_limit = adaptive_inferred_anchor_limit(analysis, config)
    query_config = replace(config, max_inferred_anchor_files=anchor_limit)
    engine = ContextEngine(config=query_config)
    workspace = query_config.workspace_root

    grep = engine.retriever.search(
        query, workspace, query_config.max_grep_candidates
    )
    discovered = discover_anchor_paths(
        query, analysis, workspace, grep, query_config
    )
    must_include_estimate = estimate_must_include_tokens(
        discovered,
        workspace,
        analysis.signals.mentioned_symbols,
    )

    min_recall = query_config.min_label_recall
    ceiling_factor = query_config.label_hard_ceiling_factor
    sweep_trace: list[dict[str, Any]] = []

    for budget in BUDGET_BUCKETS:
        bundle = engine.get_context_bundle(query, max_tokens=budget)
        paths = [c.path for c in bundle.chunks]
        recall = relevant_file_recall(expected_anchors, paths)
        tokens = sum(c.tokens for c in bundle.chunks)
        recall_ok = recall >= min_recall
        token_ok = tokens <= int(budget * ceiling_factor)
        sweep_trace.append(
            {
                "budget": budget,
                "recall_ok": recall_ok,
                "token_ok": token_ok,
                "recall": round(recall, 4),
                "tokens_used": tokens,
                "files": sorted({c.path for c in bundle.chunks}),
                "discovered_anchors": list(discovered),
            }
        )

    max_recall = max(step["recall"] for step in sweep_trace)
    recall_threshold = min(min_recall, max_recall)
    recall_candidates = [
        step for step in sweep_trace if step["recall"] >= recall_threshold - 1e-6
    ]
    if not recall_candidates:
        last = engine.get_context_bundle(query, max_tokens=BUDGET_BUCKETS[-1])
        expanded = engine.expand_context(last.bundle_id)
        last_paths = [c.path for c in last.chunks]
        last_recall = relevant_file_recall(expected_anchors, last_paths)
        return {
            "y": BUDGET_BUCKETS[-1],
            "needed_expand": True,
            "expanded_tokens": sum(c.tokens for c in expanded.chunks),
            "sweep_trace": sweep_trace,
            "discovered_anchors": discovered,
            "must_include_token_estimate": must_include_estimate,
            "relevant_file_recall_at_y": round(last_recall, 4),
            "tokens_at_y": sum(c.tokens for c in last.chunks),
            "adaptive_anchor_limit": anchor_limit,
        }

    base_tokens = int(recall_candidates[0]["tokens_used"])
    query_tokens = len(query.split())
    target_tokens = max(
        base_tokens,
        must_include_estimate,
        len(discovered) * 300,
        query_tokens * 30,
    )
    y = snap_to_bucket(target_tokens)
    min_y = BUDGET_BUCKETS[-1]
    for step in sweep_trace:
        if (
            step["recall"] >= recall_threshold - 1e-6
            and step["tokens_used"] <= int(step["budget"] * ceiling_factor)
        ):
            min_y = int(step["budget"])
            break
    y = max(y, min_y)

    tokens_at_y = next(
        (int(step["tokens_used"]) for step in sweep_trace if step["budget"] == y),
        base_tokens,
    )
    recall_at_y = next(
        (float(step["recall"]) for step in sweep_trace if step["budget"] == y),
        max_recall,
    )
    return {
        "y": y,
        "needed_expand": False,
        "sweep_trace": sweep_trace,
        "discovered_anchors": discovered,
        "must_include_token_estimate": must_include_estimate,
        "relevant_file_recall_at_y": round(recall_at_y, 4),
        "tokens_at_y": tokens_at_y,
        "adaptive_anchor_limit": anchor_limit,
    }


def sweep_one_query_inferred(
    query: str,
    engine: ContextEngine,
    config: Config,
) -> dict[str, Any]:
    """Find the smallest budget bucket that preserves inferred-anchor recall."""
    workspace = config.workspace_root
    analysis = analyze(query, config)
    grep = engine.retriever.search(
        query, workspace, config.max_grep_candidates
    )
    discovered = discover_anchor_paths(
        query, analysis, workspace, grep, config
    )
    must_include_estimate = estimate_must_include_tokens(
        discovered,
        workspace,
        analysis.signals.mentioned_symbols,
    )

    sweep_trace: list[dict[str, Any]] = []
    for budget in BUDGET_BUCKETS:
        bundle = engine.get_context_bundle(query, max_tokens=budget)
        ok = not discovered or anchors_present(bundle, discovered)
        sweep_trace.append(
            {
                "budget": budget,
                "anchors_ok": ok,
                "tokens_used": sum(c.tokens for c in bundle.chunks),
                "files": sorted({c.path for c in bundle.chunks}),
                "discovered_anchors": list(discovered),
            }
        )
        if ok:
            return {
                "y": budget,
                "needed_expand": False,
                "sweep_trace": sweep_trace,
                "discovered_anchors": discovered,
                "must_include_token_estimate": must_include_estimate,
            }

    last = engine.get_context_bundle(query, max_tokens=BUDGET_BUCKETS[-1])
    expanded = engine.expand_context(last.bundle_id)
    return {
        "y": BUDGET_BUCKETS[-1],
        "needed_expand": True,
        "expanded_tokens": sum(c.tokens for c in expanded.chunks),
        "sweep_trace": sweep_trace,
        "discovered_anchors": discovered,
        "must_include_token_estimate": must_include_estimate,
    }


def load_queries(path: Path) -> list[dict[str, Any]]:
    return load_benchmark_queries(path)


def tokens_at_budget(sweep_trace: list[dict[str, Any]], budget: int) -> int:
    """Return tiktoken bundle cost recorded at ``budget`` in the sweep trace."""
    for step in sweep_trace:
        if step["budget"] == budget:
            return int(step["tokens_used"])
    return int(sweep_trace[-1]["tokens_used"])


def _label_bucket(item: dict[str, Any], sweep: dict[str, Any]) -> int:
    """Choose training bucket: corpus target_budget or recall-aware spread."""
    if item.get("target_budget") is not None:
        return int(item["target_budget"])
    y = int(sweep["y"])
    if y < BUDGET_BUCKETS[-1]:
        return y
    discovered = sweep.get("discovered_anchors") or []
    must_include = int(sweep.get("must_include_token_estimate") or 0)
    return snap_to_bucket(int(must_include * 1.2 + len(discovered) * 200))


def label_all(workspace: Path, queries_path: Path) -> list[dict[str, Any]]:
    config = Config(workspace_root=workspace.resolve())

    rows: list[dict[str, Any]] = []

    for item in load_queries(queries_path):
        query = item["query"]
        analysis = analyze(query, config)
        expected_anchors = [str(p) for p in item.get("expected_anchors", [])]
        sweep = sweep_one_query_quality(query, config, expected_anchors)
        discovered = sweep["discovered_anchors"]
        y = _label_bucket(item, sweep)
        label_source = "quality_sweep"
        expected_tokens = int(sweep.get("tokens_at_y", tokens_at_budget(sweep["sweep_trace"], y)))
        rows.append(
            {
                "query_id": item["id"],
                "query": query,
                "intent": analysis.intent.value,
                "y": y,
                "expected_tokens": expected_tokens,
                "needed_expand": sweep["needed_expand"],
                "label_source": label_source,
                "relevant_file_recall_at_y": sweep.get("relevant_file_recall_at_y"),
                "oracle_anchor_recall": round(
                    oracle_anchor_recall(expected_anchors, discovered), 4
                ),
                "discovered_anchors": discovered,
                "adaptive_anchor_limit": sweep.get("adaptive_anchor_limit"),
                "anchor_budget": analysis.budget.recommended,
                "features": extract_features(
                    query,
                    analysis,
                    config,
                    discovered_anchor_count=len(discovered),
                    must_include_token_estimate=sweep["must_include_token_estimate"],
                ),
                "sweep_trace": sweep["sweep_trace"],
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML budget labels.")
    parser.add_argument("--workspace", type=Path, default=_DEFAULT_WORKSPACE)
    parser.add_argument("--queries", type=Path, default=_DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = label_all(args.workspace, args.queries)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} labels to {args.output}")


if __name__ == "__main__":
    main()
