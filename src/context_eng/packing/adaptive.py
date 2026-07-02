"""Adaptive optional chunk cap for budget packing."""

from __future__ import annotations

_BASE_CAP = 2
_BUDGET_BOOST_THRESHOLD = 8000
_ANCHOR_BOOST_THRESHOLD = 4
_QUERY_TOKENS_BOOST_THRESHOLD = 40
_LOW_BUDGET_CAP = 6000


def adaptive_max_optional_chunks(
    *,
    budget_limit: int,
    anchor_count: int,
    query_tokens: int,
    floor: int = 1,
    upper: int = 4,
) -> int:
    """Compute per-query optional chunk cap from budget, anchors, and query size."""
    cap = _BASE_CAP
    if budget_limit >= _BUDGET_BOOST_THRESHOLD:
        cap += 1
    if anchor_count >= _ANCHOR_BOOST_THRESHOLD:
        cap += 1
    if query_tokens >= _QUERY_TOKENS_BOOST_THRESHOLD:
        cap += 1
    if budget_limit < _LOW_BUDGET_CAP:
        cap = min(cap, _BASE_CAP)
    return max(floor, min(cap, upper))
