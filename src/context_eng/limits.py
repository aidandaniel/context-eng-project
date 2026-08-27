"""Hard input limits so a single MCP call cannot unbounded-grow context or CPU.

Keep this module free of Config / engine imports to avoid cycles.
Values stay aligned with ``BUDGET_BUCKETS`` in ``ml.budget_model``.
"""

from __future__ import annotations

MAX_QUERY_CHARS = 16_384
MAX_EXPANSIONS = 3
MIN_BUDGET_TOKENS = 2000
MAX_BUDGET_TOKENS = 15_000


def validate_query(query: str, max_chars: int = MAX_QUERY_CHARS) -> str:
    """Return a stripped query or raise ``ValueError``."""
    if query is None:
        raise ValueError("query is required")
    text = str(query).strip()
    if not text:
        raise ValueError("query must be non-empty")
    if len(text) > max_chars:
        raise ValueError(f"query exceeds {max_chars} characters")
    return text
