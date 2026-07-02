"""Resolve runtime token budget from trained RF model (with explicit override)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from context_eng.config import Config
from context_eng.ml.budget_model import RandomForestBudgetModel, snap_to_bucket
from context_eng.ml.features import extract_features
from context_eng.models import QueryAnalysis

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL = _REPO_ROOT / "ml" / "models" / "budget_rf_v2.joblib"


@dataclass(frozen=True)
class BudgetResolution:
    """Resolved token ceiling and how it was chosen."""

    limit: int
    source: str  # explicit | rf | fallback_default


@lru_cache(maxsize=4)
def _load_budget_model(model_path: str) -> RandomForestBudgetModel:
    return RandomForestBudgetModel.load(model_path)


def default_model_path(config: Config) -> Path:
    if config.ml_model_path is not None:
        return Path(config.ml_model_path)
    return _DEFAULT_MODEL


def rf_budget(query: str, analysis: QueryAnalysis, config: Config) -> int:
    """Predict a budget bucket using the trained Random Forest model."""
    model_path = default_model_path(config)
    if not model_path.is_file():
        raise FileNotFoundError(f"RF budget model not found: {model_path}")
    model = _load_budget_model(str(model_path.resolve()))
    features = extract_features(query, analysis, config)
    return model.predict(features).budget


def resolve_budget(
    query: str,
    analysis: QueryAnalysis,
    config: Config,
    max_tokens: int | None,
) -> BudgetResolution:
    """Pick the token ceiling for ``get_context_bundle``."""
    if max_tokens is not None:
        return BudgetResolution(max_tokens, "explicit")
    try:
        return BudgetResolution(rf_budget(query, analysis, config), "rf")
    except FileNotFoundError:
        limit = snap_to_bucket(config.default_max_tokens)
        return BudgetResolution(limit, "fallback_default")


def resolve_budget_limit(
    query: str,
    analysis: QueryAnalysis,
    config: Config,
    max_tokens: int | None,
) -> int:
    """Return only the resolved token ceiling (see ``resolve_budget`` for source)."""
    return resolve_budget(query, analysis, config, max_tokens).limit
