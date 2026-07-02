"""Tests for adaptive optional chunk cap (MCP v2 phase 5)."""

from context_eng.config import Config
from context_eng.packing.adaptive import adaptive_max_optional_chunks


def test_adaptive_cap_base_narrow_query():
    cap = adaptive_max_optional_chunks(
        budget_limit=4000,
        anchor_count=1,
        query_tokens=10,
        floor=1,
        upper=4,
    )
    assert cap == 2


def test_adaptive_cap_boosts_for_broad_query():
    cap = adaptive_max_optional_chunks(
        budget_limit=10000,
        anchor_count=5,
        query_tokens=50,
        floor=1,
        upper=4,
    )
    assert cap == 4


def test_adaptive_cap_low_budget_clamps_boosts():
    cap = adaptive_max_optional_chunks(
        budget_limit=5000,
        anchor_count=6,
        query_tokens=60,
        floor=1,
        upper=4,
    )
    assert cap == 2


def test_adaptive_cap_respects_floor_and_upper():
    cap = adaptive_max_optional_chunks(
        budget_limit=10000,
        anchor_count=0,
        query_tokens=0,
        floor=3,
        upper=3,
    )
    assert cap == 3


def test_config_defaults_use_adaptive_not_fixed_two():
    cfg = Config(workspace_root=".")
    assert cfg.max_optional_chunks is None
    assert cfg.max_optional_chunks_upper == 4
    assert cfg.max_optional_chunks_floor == 1
