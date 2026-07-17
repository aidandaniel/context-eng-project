"""Tests for adaptive optional chunk cap (MCP v2 phase 5).

Overview
========
The adaptive optional chunk cap dynamically determines how many non-anchor ("optional")
chunks should be included in a packing decision based on:
  - Available token budget
  - Number of anchor chunks (high-confidence includes)
  - Query complexity (token count)

This adaptive approach allows the system to prioritize optional chunks when conditions
are favorable (large budget, broad query, many anchors) while being conservative when
budget is tight.

Algorithm
=========
The `adaptive_max_optional_chunks()` function computes a cap from these thresholds:

1. Base cap: 2 optional chunks
2. Budget boost: +1 if budget >= 8000 tokens
3. Anchor boost: +1 if anchor_count >= 4
4. Query boost: +1 if query_tokens >= 40
5. Low-budget clamp: cap resets to 2 if budget < 6000 tokens
6. Bounds: final cap is clamped to [floor, upper]

The result is a simple integer that bounds how many optional chunks to consider,
regardless of whether budget remains after including anchors.

Configuration
=============
By default, Config.max_optional_chunks is None, which enables adaptive mode:
  - max_optional_chunks_upper: 4 (hard maximum)
  - max_optional_chunks_floor: 1 (hard minimum)

When max_optional_chunks is set to a fixed integer, adaptive mode is disabled
and the fixed value is used instead (legacy behavior).

Test Coverage
=============
These tests verify:
  - Base behavior: narrow queries get conservative cap (2)
  - Broad query boost: high budget + many anchors + large query get maximum cap (4)
  - Low-budget clamp: even with many anchors/query tokens, tight budget keeps cap low (2)
  - Bound respect: floor and upper are always enforced
  - Config defaults: ensure new Config instances use adaptive mode by default
"""

from context_eng.config import Config
from context_eng.packing.adaptive import adaptive_max_optional_chunks


def test_adaptive_cap_base_narrow_query():
    """Narrow query with modest budget gets base cap of 2."""
    cap = adaptive_max_optional_chunks(
        budget_limit=4000,
        anchor_count=1,
        query_tokens=10,
        floor=1,
        upper=4,
    )
    assert cap == 2


def test_adaptive_cap_boosts_for_broad_query():
    """Broad query with high budget and many anchors gets maximum cap of 4.
    
    Boost logic:
      - Base: 2
      - Budget boost: 10000 >= 8000 → +1 = 3
      - Anchor boost: 5 >= 4 → +1 = 4
      - Query boost: 50 >= 40 → +1 = 5, but clamped to upper=4
    """
    cap = adaptive_max_optional_chunks(
        budget_limit=10000,
        anchor_count=5,
        query_tokens=50,
        floor=1,
        upper=4,
    )
    assert cap == 4


def test_adaptive_cap_low_budget_clamps_boosts():
    """Even with many anchors and large query, low budget forces conservative cap.
    
    The low-budget clamp (< 6000 tokens) resets cap to base (2) regardless of boosts:
      - Base: 2
      - Budget boost: 5000 < 8000 → no boost
      - Anchor boost: 6 >= 4 → +1 = 3
      - Query boost: 60 >= 40 → +1 = 4
      - Low-budget clamp: 5000 < 6000 → cap = min(4, 2) = 2
    """
    cap = adaptive_max_optional_chunks(
        budget_limit=5000,
        anchor_count=6,
        query_tokens=60,
        floor=1,
        upper=4,
    )
    assert cap == 2


def test_adaptive_cap_respects_floor_and_upper():
    """Floor and upper bounds are always enforced.
    
    With no query, anchors, or budget boosts, the raw cap would be 2.
    But when floor=3 and upper=3, the result is clamped to exactly 3.
    """
    cap = adaptive_max_optional_chunks(
        budget_limit=10000,
        anchor_count=0,
        query_tokens=0,
        floor=3,
        upper=3,
    )
    assert cap == 3


def test_config_defaults_use_adaptive_not_fixed_two():
    """New Config instances default to adaptive mode, not a fixed cap.
    
    This verifies:
      - max_optional_chunks is None (enables adaptive mode)
      - max_optional_chunks_upper defaults to 4
      - max_optional_chunks_floor defaults to 1
    """
    cfg = Config(workspace_root=".")
    assert cfg.max_optional_chunks is None
    assert cfg.max_optional_chunks_upper == 4
    assert cfg.max_optional_chunks_floor == 1
