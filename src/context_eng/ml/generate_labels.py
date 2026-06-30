"""Generate budget-bucket labels from the RF training corpus."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.intent.classifier import analyze
from context_eng.ml.budget_model import BUDGET_BUCKETS
from context_eng.ml.features import extract_features
from benchmarks.query_loader import load_queries as load_benchmark_queries

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WORKSPACE = _REPO_ROOT / "benchmarks" / "fixture_repo"
_DEFAULT_QUERIES = _REPO_ROOT / "ml" / "data" / "budget_training_queries.yaml"
_DEFAULT_OUTPUT = _REPO_ROOT / "ml" / "data" / "budget_labels.jsonl"
_TRAINING_CONFIG_OVERRIDES = {
    "max_grep_candidates": 200,
    "max_inferred_anchor_files": 40,
    "inferred_anchor_min_score": 0.0,
    "max_optional_chunks": 16,
    "min_chunk_score": 0.0,
}


def anchors_present(bundle: Any, expected_anchors: list[str]) -> bool:
    """Return whether every expected anchor is represented in ``bundle``."""
    paths = {c.path for c in bundle.chunks}
    for anchor in expected_anchors:
        normalized = anchor.replace("\\", "/")
        if not any(p == normalized or p.endswith(normalized) for p in paths):
            return False
    return True


def sweep_one_query(
    query: str,
    expected_anchors: list[str],
    engine: ContextEngine,
) -> dict[str, Any]:
    """Find the smallest budget bucket that preserves anchor recall."""
    sweep_trace: list[dict[str, Any]] = []
    for budget in BUDGET_BUCKETS:
        bundle = engine.get_context_bundle(query, max_tokens=budget)
        ok = anchors_present(bundle, expected_anchors)
        sweep_trace.append(
            {
                "budget": budget,
                "anchors_ok": ok,
                "tokens_used": sum(c.tokens for c in bundle.chunks),
                "files": sorted({c.path for c in bundle.chunks}),
            }
        )
        if ok:
            return {
                "y": budget,
                "needed_expand": False,
                "sweep_trace": sweep_trace,
            }

    last = engine.get_context_bundle(query, max_tokens=BUDGET_BUCKETS[-1])
    expanded = engine.expand_context(last.bundle_id)
    return {
        "y": BUDGET_BUCKETS[-1],
        "needed_expand": True,
        "expanded_tokens": sum(c.tokens for c in expanded.chunks),
        "sweep_trace": sweep_trace,
    }


def load_queries(path: Path) -> list[dict[str, Any]]:
    return load_benchmark_queries(path)


def tokens_at_budget(sweep_trace: list[dict[str, Any]], budget: int) -> int:
    """Return tiktoken bundle cost recorded at ``budget`` in the sweep trace."""
    for step in sweep_trace:
        if step["budget"] == budget:
            return int(step["tokens_used"])
    return int(sweep_trace[-1]["tokens_used"])


def label_all(workspace: Path, queries_path: Path) -> list[dict[str, Any]]:
    config = Config(workspace_root=workspace.resolve())
    config = replace(config, **_TRAINING_CONFIG_OVERRIDES)
    engine = ContextEngine(config=config)
    rows: list[dict[str, Any]] = []

    for item in load_queries(queries_path):
        query = item["query"]
        analysis = analyze(query, config)
        sweep = sweep_one_query(query, item.get("expected_anchors", []), engine)
        target_budget = item.get("target_budget")
        label_source = "sweep"
        y = sweep["y"]
        if target_budget is not None:
            target = int(target_budget)
            if target not in BUDGET_BUCKETS:
                raise ValueError(
                    f"target_budget for {item['id']} must be one of {BUDGET_BUCKETS}"
                )
            ok_at_target = any(
                step["budget"] == target and step["anchors_ok"]
                for step in sweep["sweep_trace"]
            )
            if ok_at_target:
                y = target
                label_source = "target_budget"
        expected_tokens = tokens_at_budget(sweep["sweep_trace"], y)
        rows.append(
            {
                "query_id": item["id"],
                "query": query,
                "intent": analysis.intent.value,
                "y": y,
                "expected_tokens": expected_tokens,
                "needed_expand": sweep["needed_expand"],
                "label_source": label_source,
                "anchor_budget": analysis.budget.recommended,
                "features": extract_features(query, analysis, config),
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
