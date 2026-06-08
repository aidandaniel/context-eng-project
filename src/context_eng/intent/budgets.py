"""Fixed intent -> token budget table with clamp bounds.

This is the rule-based baseline. The shape (recommended/min/max + source) is
intentionally compatible with a future ML-predicted budget so callers do not
change when ml_v1 lands.
"""

from __future__ import annotations

from context_eng.config import DEFAULT_INTENT_BUDGETS, Config
from context_eng.models import BudgetInfo, Intent


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def budget_for(intent: Intent, config: Config | None = None) -> BudgetInfo:
    """Return the fixed BudgetInfo for ``intent``."""
    table = config.intent_budgets if config is not None else DEFAULT_INTENT_BUDGETS
    recommended, low, high = table.get(intent.value, table["implement"])
    return BudgetInfo(
        recommended=recommended,
        min=low,
        max=high,
        source="fixed_intent_table",
    )
