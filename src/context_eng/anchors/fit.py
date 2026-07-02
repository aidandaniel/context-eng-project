"""Budget auto-fit: ensure the RF token ceiling can fit discovered anchors."""

from __future__ import annotations

from pathlib import Path

from context_eng.ml.budget_model import BUDGET_BUCKETS
from context_eng.retrieval.symbol_slice import find_symbol_span
from context_eng.tokens.estimator import count_tokens
from context_eng.workspace import read_text

# Must match engine._HEAD_LINES for consistent token estimates.
_HEAD_LINES = 40


def estimate_must_include_tokens(
    anchor_paths: list[str],
    workspace: Path,
    symbols: list[str] | None = None,
) -> int:
    """Estimate tokens required for must-include anchor chunks."""
    symbols = symbols or []
    total = 0
    for rel in anchor_paths:
        path = workspace / rel
        source = read_text(path)
        if not source:
            continue
        lines = source.splitlines()
        snippet_tokens = 0
        for sym in symbols:
            span = find_symbol_span(source, path.name, sym)
            if span is not None:
                snippet = "\n".join(lines[span.start_line - 1 : span.end_line])
                snippet_tokens = max(snippet_tokens, count_tokens(snippet))
        if snippet_tokens == 0:
            end = min(len(lines), _HEAD_LINES)
            snippet = "\n".join(lines[:end])
            snippet_tokens = count_tokens(snippet)
        total += snippet_tokens
    return total


def ensure_budget_fits_anchors(
    budget_limit: int,
    anchor_paths: list[str],
    workspace: Path,
    symbols: list[str] | None = None,
    *,
    hard_ceiling_factor: float = 1.5,
) -> int:
    """Bump ``budget_limit`` along ``BUDGET_BUCKETS`` when anchors exceed the ceiling."""
    if not anchor_paths:
        return budget_limit

    estimated = estimate_must_include_tokens(anchor_paths, workspace, symbols)
    fitted = budget_limit
    while True:
        ceiling = int(fitted * hard_ceiling_factor)
        if estimated <= ceiling:
            return fitted
        next_bucket: int | None = None
        for bucket in BUDGET_BUCKETS:
            if bucket > fitted:
                next_bucket = bucket
                break
        if next_bucket is None:
            return BUDGET_BUCKETS[-1]
        fitted = next_bucket
