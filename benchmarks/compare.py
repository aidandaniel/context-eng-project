"""Before/after benchmark: quantify token reduction and latency.

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

from benchmarks.baseline import run_baseline
from benchmarks.query_loader import load_queries
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
    latency_ms: float
    inferred_files: list[str]


def run_benchmark(
    workspace: Path,
    queries: list[dict],
    config: Config | None = None,
) -> list[QueryReport]:
    cfg = config or Config(workspace_root=workspace.resolve())
    engine = ContextEngine(config=cfg)

    reports: list[QueryReport] = []
    for q in queries:
        baseline = run_baseline(
            q["query"], config, q.get("baseline_strategy", "grep_top_k_full_files"),
            int(q.get("baseline_k", 5)),
        )
        mcp = run_mcp(q["query"], engine)

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
                latency_ms=mcp.latency_ms,
                inferred_files=mcp.files,
            )
        )
    return reports


def aggregate(reports: list[QueryReport]) -> dict:
    reductions = [r.reduction_pct for r in reports]
    latencies = sorted(r.latency_ms for r in reports)
    p90_idx = max(0, int(len(latencies) * 0.9) - 1) if latencies else 0
    return {
        "query_count": len(reports),
        "median_reduction_pct": round(statistics.median(reductions), 1) if reductions else 0.0,
        "mean_reduction_pct": round(statistics.fmean(reductions), 1) if reductions else 0.0,
        "median_baseline_tokens": int(statistics.median(r.baseline_tokens for r in reports)) if reports else 0,
        "median_mcp_tokens": int(statistics.median(r.mcp_tokens for r in reports)) if reports else 0,
        "p90_latency_ms": latencies[p90_idx] if latencies else 0.0,
    }


def _format_markdown(reports: list[QueryReport], agg: dict) -> str:
    lines = [
        "# Context Engineering MCP - Benchmark Report",
        "",
        f"Queries: {agg['query_count']}  |  "
        f"Median reduction: {agg['median_reduction_pct']}%  |  "
        f"p90 latency: {agg['p90_latency_ms']} ms",
        "",
        "| Query | Intent | Baseline | MCP | Reduction | Inferred files |",
        "|-------|--------|---------:|----:|----------:|:--------------|",
    ]
    for r in reports:
        lines.append(
            f"| {r.id} | {r.intent} | {r.baseline_tokens:,} | {r.mcp_tokens:,} | "
            f"{r.reduction_pct}% | {', '.join(r.inferred_files) or '-'} |"
        )
    lines.append(
        f"| **MEDIAN** | - | {agg['median_baseline_tokens']:,} | "
        f"{agg['median_mcp_tokens']:,} | {agg['median_reduction_pct']}% | - |"
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


def evaluate(
    workspace: Path,
    queries_path: Path,
    output: Path,
    config: Config | None = None,
) -> dict:
    queries = load_queries(queries_path)
    reports = run_benchmark(workspace, queries, config=config)
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
