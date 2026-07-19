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
# Benchmark defaults to the SWE-bench Lite–trained budget RF.
_DEFAULT_MODEL = _REPO_ROOT / "ml" / "models" / "budget_rf_swebench.joblib"


@dataclass
class QueryReport:
    id: str
    intent: str
    baseline_tokens: int
    mcp_tokens: int
    reduction_pct: float
    latency_ms: float
    inferred_files: list[str]


def benchmark_config(
    workspace: Path,
    *,
    model_path: Path | None = None,
) -> Config:
    """Config for benchmark runs; points RF budget at the SWE-bench model."""
    return Config(
        workspace_root=workspace.resolve(),
        ml_model_path=(model_path or _DEFAULT_MODEL).resolve(),
    )


def run_benchmark(
    workspace: Path,
    queries: list[dict],
    config: Config | None = None,
) -> list[QueryReport]:
    cfg = config or benchmark_config(workspace)
    engine = ContextEngine(config=cfg)

    reports: list[QueryReport] = []
    for q in queries:
        baseline = run_baseline(
            q["query"], cfg, q.get("baseline_strategy", "grep_top_k_full_files"),
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


def aggregate(reports: list[QueryReport], *, model_path: Path | None = None) -> dict:
    reductions = [r.reduction_pct for r in reports]
    latencies = sorted(r.latency_ms for r in reports)
    p90_idx = max(0, int(len(latencies) * 0.9) - 1) if latencies else 0
    payload = {
        "query_count": len(reports),
        "median_reduction_pct": round(statistics.median(reductions), 1) if reductions else 0.0,
        "mean_reduction_pct": round(statistics.fmean(reductions), 1) if reductions else 0.0,
        "median_baseline_tokens": int(statistics.median(r.baseline_tokens for r in reports)) if reports else 0,
        "median_mcp_tokens": int(statistics.median(r.mcp_tokens for r in reports)) if reports else 0,
        "p90_latency_ms": latencies[p90_idx] if latencies else 0.0,
    }
    if model_path is not None:
        payload["ml_model_path"] = str(model_path)
    return payload


def _format_markdown(reports: list[QueryReport], agg: dict) -> str:
    lines = [
        "# Context Engineering MCP - Benchmark Report",
        "",
    ]
    if agg.get("ml_model_path"):
        lines.extend([f"Model: `{agg['ml_model_path']}`", ""])
    lines.extend(
        [
            f"Queries: {agg['query_count']}  |  "
            f"Median reduction: {agg['median_reduction_pct']}%  |  "
            f"p90 latency: {agg['p90_latency_ms']} ms",
            "",
            "| Query | Intent | Baseline | MCP | Reduction | Inferred files |",
            "|-------|--------|---------:|----:|----------:|:--------------|",
        ]
    )
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
    cfg = config or benchmark_config(workspace)
    queries = load_queries(queries_path)
    reports = run_benchmark(workspace, queries, config=cfg)
    agg = aggregate(reports, model_path=cfg.ml_model_path)
    write_reports(reports, agg, output)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the context-eng benchmark.")
    parser.add_argument("--workspace", type=Path, default=_DEFAULT_WORKSPACE)
    parser.add_argument("--queries", type=Path, default=_DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=_DEFAULT_MODEL,
        help="RF budget model joblib (default: ml/models/budget_rf_swebench.joblib)",
    )
    args = parser.parse_args()

    cfg = benchmark_config(args.workspace, model_path=args.model_path)
    agg = evaluate(args.workspace, args.queries, args.output, config=cfg)

    print("Context Engineering MCP - Benchmark Summary")
    print("-" * 44)
    for key, value in agg.items():
        print(f"{key:>24}: {value}")
    print("-" * 44)
    print(f"Reports written under: {args.output}.{{json,md}}")


if __name__ == "__main__":
    main()
