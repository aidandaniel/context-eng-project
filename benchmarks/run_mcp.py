"""Run the MCP path (after) for a single benchmark query.

Mirrors realistic agent behavior: analyze, infer anchors from the query when
possible, and get a budgeted bundle. The benchmark no longer uses hidden gold
labels or oracle-driven expansion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from context_eng.engine import ContextEngine


@dataclass
class McpResult:
    tokens: int
    files: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    bundle_id: str = ""


def run_mcp(
    query: str,
    engine: ContextEngine,
) -> McpResult:
    start = time.perf_counter()
    bundle = engine.get_context_bundle(query)

    latency_ms = (time.perf_counter() - start) * 1000.0
    tokens = sum(c.tokens for c in bundle.chunks)
    return McpResult(
        tokens=tokens,
        files=sorted({c.path for c in bundle.chunks}),
        latency_ms=round(latency_ms, 2),
        bundle_id=bundle.bundle_id,
    )
