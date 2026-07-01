"""Quick sweep of config knobs vs median token reduction."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from benchmarks.compare import aggregate, run_benchmark
from benchmarks.query_loader import load_queries
from context_eng.config import Config

ws = Path("benchmarks/fixture_repo")
queries = load_queries(Path("benchmarks/queries.yaml"))
base = Config(workspace_root=ws.resolve(), budget_source="rf")

variants: list[tuple[str, Config]] = [
    ("default", base),
    ("min_score_0.25", replace(base, min_chunk_score=0.25)),
    ("min_score_0.30", replace(base, min_chunk_score=0.30)),
    ("opt_2", replace(base, max_optional_chunks=2)),
    ("opt_2_score_0.25", replace(base, max_optional_chunks=2, min_chunk_score=0.25)),
    ("opt_3_score_0.20", replace(base, max_optional_chunks=3, min_chunk_score=0.20)),
    ("opt_2_score_0.30", replace(base, max_optional_chunks=2, min_chunk_score=0.30)),
]

for label, cfg in variants:
    agg = aggregate(run_benchmark(ws, queries, cfg))
    print(
        f"{label:22} {agg['median_reduction_pct']:5.1f}%  "
        f"mcp={agg['median_mcp_tokens']}  p90={agg['p90_latency_ms']:.0f}ms"
    )
