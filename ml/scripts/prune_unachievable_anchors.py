"""Trim expected_anchors to files retrievable at each query's budget ceiling."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from benchmarks.query_loader import load_queries
from context_eng.config import Config
from context_eng.engine import ContextEngine

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QUERIES = _REPO_ROOT / "ml" / "data" / "budget_training_queries.yaml"
_MAX_BUDGET = 15000
_HEADER = (
    "# RF training corpus for budget-bucket labels.\n"
    "# The query text stays natural-language and path-free; expected_anchors are\n"
    "# hidden labels used only for label generation.\n\n"
)
_TRAINING_OVERRIDES = {
    "max_grep_candidates": 200,
    "max_inferred_anchor_files": 40,
    "inferred_anchor_min_score": 0.0,
    "max_optional_chunks": 16,
    "min_chunk_score": 0.0,
}


def _format_entry(item: dict) -> str:
    lines = [f"- id: {item['id']}", f'  query: "{item["query"]}"']
    anchors = item.get("expected_anchors", [])
    anchor_parts = ", ".join(f'"{a}"' for a in anchors)
    lines.append(f"  expected_anchors: [{anchor_parts}]")
    if item.get("target_budget") is not None:
        lines.append(f"  target_budget: {int(item['target_budget'])}")
    return "\n".join(lines)


def _achievable_anchors(
    engine: ContextEngine,
    query: str,
    anchors: list[str],
    budget: int,
) -> list[str]:
    bundle = engine.get_context_bundle(query, max_tokens=budget)
    paths = {chunk.path for chunk in bundle.chunks}
    kept: list[str] = []
    for anchor in anchors:
        normalized = anchor.replace("\\", "/")
        if any(path == normalized or path.endswith(normalized) for path in paths):
            kept.append(anchor)
    return kept


def prune_queries(queries_path: Path, workspace: Path) -> tuple[int, int]:
    config = Config(workspace_root=workspace.resolve())
    config = replace(config, **_TRAINING_OVERRIDES)
    engine = ContextEngine(config=config)

    items = load_queries(queries_path)
    trimmed = 0
    for item in items:
        anchors = list(item.get("expected_anchors", []))
        if not anchors:
            continue
        budget = int(item.get("target_budget") or _MAX_BUDGET)
        kept = _achievable_anchors(engine, item["query"], anchors, budget)
        if kept != anchors:
            item["expected_anchors"] = kept
            trimmed += 1

    body = "\n\n".join(_format_entry(item) for item in items) + "\n"
    queries_path.write_text(_HEADER + body, encoding="utf-8")
    return trimmed, len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune unachievable training anchors.")
    parser.add_argument("--queries", type=Path, default=_DEFAULT_QUERIES)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "fixture_repo",
    )
    args = parser.parse_args()
    trimmed, total = prune_queries(args.queries, args.workspace)
    print(f"Pruned anchors on {trimmed}/{total} queries -> {args.queries}")


if __name__ == "__main__":
    main()
