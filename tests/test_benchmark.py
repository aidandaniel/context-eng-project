"""Benchmark gate.

Runs the full before/after benchmark on the fixture repo and asserts the MVP
success criteria. Marked ``benchmark`` so it can be selected or skipped:

    pytest -m benchmark            # run only this gate
    pytest -m "not benchmark"      # skip it (fast unit run)
"""

from pathlib import Path

import pytest

from benchmarks.compare import aggregate, load_queries, run_benchmark

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _REPO_ROOT / "benchmarks" / "fixture_repo"
_QUERIES = _REPO_ROOT / "benchmarks" / "queries.yaml"

# Gate thresholds (see plan "Success criteria for MVP").
MIN_MEDIAN_REDUCTION_PCT = 30.0
MAX_P90_LATENCY_MS = 3000.0


@pytest.fixture(scope="module")
def results():
    queries = load_queries(_QUERIES)
    reports = run_benchmark(_WORKSPACE, queries)
    return reports, aggregate(reports)


@pytest.mark.benchmark
def test_median_token_reduction(results):
    _, agg = results
    assert agg["median_reduction_pct"] >= MIN_MEDIAN_REDUCTION_PCT, (
        f"median reduction {agg['median_reduction_pct']}% "
        f"below gate {MIN_MEDIAN_REDUCTION_PCT}%"
    )


@pytest.mark.benchmark
def test_p90_latency(results):
    _, agg = results
    assert agg["p90_latency_ms"] < MAX_P90_LATENCY_MS, (
        f"p90 latency {agg['p90_latency_ms']}ms exceeds {MAX_P90_LATENCY_MS}ms"
    )


@pytest.mark.benchmark
def test_reproducible(results):
    """A second run should produce the same median reduction (deterministic)."""
    queries = load_queries(_QUERIES)
    agg2 = aggregate(run_benchmark(_WORKSPACE, queries))
    _, agg1 = results
    assert abs(agg1["median_reduction_pct"] - agg2["median_reduction_pct"]) < 5.0
