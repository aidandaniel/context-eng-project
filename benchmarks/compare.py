"""Before/after benchmark: quantify token reduction and anchor recall.

Runs every query in queries.yaml through the baseline (full-file) strategy and
the MCP (budgeted bundle) strategy, then writes a JSON + Markdown report and
prints an aggregate summary.

Usage:
    context-eng-benchmark --workspace benchmarks/fixture_repo \
        --queries benchmarks/queries.yaml --output benchmarks/results/latest
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from benchmarks.baseline import run_baseline
from benchmarks.run_mcp import run_mcp
from context_eng.config import Config
from context_eng.engine import ContextEngine

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WORKSPACE = _REPO_ROOT / "benchmarks" / "fixture_repo"
_DEFAULT_QUERIES = _REPO_ROOT / "benchmarks" / "queries.yaml"
_DEFAULT_OUTPUT = _REPO_ROOT / "benchmarks" / "results" / "latest"


@dataclass
class QueryReport:
    id: str
    intent: str
    baseline_tokens: int
    mcp_tokens: int
    reduction_pct: float
    anchor_recall: float
    supporting_recall: float
    expansions_used: int
    latency_ms: float
    missing_anchors: list[str]


def _recall(expected: list[str], got: set[str]) -> tuple[float, list[str]]:
    if not expected:
        return 1.0, []
    missing = []
    hits = 0
    for item in expected:
        norm = item.replace("\\", "/")
        if any(p == norm or p.endswith(norm) for p in got):
            hits += 1
        else:
            missing.append(item)
    return hits / len(expected), missing


def load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_benchmark(workspace: Path, queries: list[dict]) -> list[QueryReport]:
    config = Config(workspace_root=workspace.resolve())
    # Disable event logging noise during benchmarking by pointing at results.
    engine = ContextEngine(config=config)

    reports: list[QueryReport] = []
    for q in queries:
        baseline = run_baseline(
            q["query"], config, q.get("baseline_strategy", "grep_top_k_full_files"),
            int(q.get("baseline_k", 5)),
        )
        mcp = run_mcp(q["query"], engine, q.get("expected_anchors", []))

        got = set(mcp.files)
        anchor_recall, missing = _recall(q.get("expected_anchors", []), got)
        supporting_recall, _ = _recall(q.get("expected_supporting", []), got)

        reduction = (
            (baseline.tokens - mcp.tokens) / baseline.tokens * 100.0
            if baseline.tokens
            else 0.0
        )
        reports.append(
            QueryReport(
                id=q["id"],
                intent=q.get("intent", "unknown"),
                baseline_tokens=baseline.tokens,
                mcp_tokens=mcp.tokens,
                reduction_pct=round(reduction, 1),
                anchor_recall=round(anchor_recall, 3),
                supporting_recall=round(supporting_recall, 3),
                expansions_used=mcp.expansions,
                latency_ms=mcp.latency_ms,
                missing_anchors=missing,
            )
        )
    return reports


def aggregate(reports: list[QueryReport]) -> dict:
    reductions = [r.reduction_pct for r in reports]
    anchor_recalls = [r.anchor_recall for r in reports]
    supporting = [r.supporting_recall for r in reports]
    latencies = sorted(r.latency_ms for r in reports)
    p90_idx = max(0, int(len(latencies) * 0.9) - 1) if latencies else 0
    return {
        "query_count": len(reports),
        "median_reduction_pct": round(statistics.median(reductions), 1) if reductions else 0.0,
        "mean_reduction_pct": round(statistics.fmean(reductions), 1) if reductions else 0.0,
        "median_anchor_recall": round(statistics.median(anchor_recalls), 3) if anchor_recalls else 0.0,
        "min_anchor_recall": round(min(anchor_recalls), 3) if anchor_recalls else 0.0,
        "mean_supporting_recall": round(statistics.fmean(supporting), 3) if supporting else 0.0,
        "median_baseline_tokens": int(statistics.median(r.baseline_tokens for r in reports)) if reports else 0,
        "median_mcp_tokens": int(statistics.median(r.mcp_tokens for r in reports)) if reports else 0,
        "p90_latency_ms": latencies[p90_idx] if latencies else 0.0,
        "total_expansions": sum(r.expansions_used for r in reports),
    }


def _format_markdown(reports: list[QueryReport], agg: dict) -> str:
    lines = [
        "# Context Engineering MCP - Benchmark Report",
        "",
        f"Queries: {agg['query_count']}  |  "
        f"Median reduction: {agg['median_reduction_pct']}%  |  "
        f"Median anchor recall: {agg['median_anchor_recall'] * 100:.0f}%  |  "
        f"p90 latency: {agg['p90_latency_ms']} ms",
        "",
        "| Query | Intent | Baseline | MCP | Reduction | Anchor | Support | Exp |",
        "|-------|--------|---------:|----:|----------:|:------:|:-------:|:---:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.id} | {r.intent} | {r.baseline_tokens:,} | {r.mcp_tokens:,} | "
            f"{r.reduction_pct}% | {r.anchor_recall * 100:.0f}% | "
            f"{r.supporting_recall * 100:.0f}% | {r.expansions_used} |"
        )
    lines.append(
        f"| **MEDIAN** | - | {agg['median_baseline_tokens']:,} | "
        f"{agg['median_mcp_tokens']:,} | {agg['median_reduction_pct']}% | "
        f"{agg['median_anchor_recall'] * 100:.0f}% | "
        f"{agg['mean_supporting_recall'] * 100:.0f}% | {agg['total_expansions']} |"
    )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    reports: list[QueryReport], agg: dict, output: Path
) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    md_path = output.with_suffix(".md")
    json_path.write_text(
        json.dumps(
            {"aggregate": agg, "queries": [asdict(r) for r in reports]},
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(_format_markdown(reports, agg), encoding="utf-8")
    return json_path, md_path


def evaluate(workspace: Path, queries_path: Path, output: Path) -> dict:
    queries = load_queries(queries_path)
    reports = run_benchmark(workspace, queries)
    agg = aggregate(reports)
    write_reports(reports, agg, output)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the context-eng benchmark.")
    parser.add_argument("--workspace", type=Path, default=_DEFAULT_WORKSPACE)
    parser.add_argument("--queries", type=Path, default=_DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    agg = evaluate(args.workspace, args.queries, args.output)

    print("Context Engineering MCP - Benchmark Summary")
    print("-" * 44)
    for key, value in agg.items():
        print(f"{key:>24}: {value}")
    print("-" * 44)
    print(f"Reports written under: {args.output}.{{json,md}}")


if __name__ == "__main__":
    main()
