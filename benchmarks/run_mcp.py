"""Run the MCP path (after) for a single benchmark query.

Mirrors realistic agent behavior: analyze, get a budgeted bundle, and if a
required anchor is missing, expand once (progressive disclosure).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from context_eng.engine import ContextEngine


@dataclass
class McpResult:
    tokens: int
    files: list[str] = field(default_factory=list)
    expansions: int = 0
    latency_ms: float = 0.0
    bundle_id: str = ""


def _bundle_paths(bundle) -> set[str]:
    return {c.path for c in bundle.chunks}


def _anchors_present(bundle, expected_anchors: list[str]) -> bool:
    paths = _bundle_paths(bundle)
    for anchor in expected_anchors:
        a = anchor.replace("\\", "/")
        if not any(p == a or p.endswith(a) for p in paths):
            return False
    return True


def run_mcp(
    query: str,
    engine: ContextEngine,
    expected_anchors: list[str],
) -> McpResult:
    start = time.perf_counter()
    bundle = engine.get_context_bundle(query)

    expansions = 0
    if expected_anchors and not _anchors_present(bundle, expected_anchors):
        bundle = engine.expand_context(bundle.bundle_id)
        expansions = bundle.expansions

    latency_ms = (time.perf_counter() - start) * 1000.0
    tokens = sum(c.tokens for c in bundle.chunks)
    return McpResult(
        tokens=tokens,
        files=sorted(_bundle_paths(bundle)),
        expansions=expansions,
        latency_ms=round(latency_ms, 2),
        bundle_id=bundle.bundle_id,
    )
