"""Random Forest budget-bucket prediction.

The model predicts one of the supported token ceilings rather than a raw token
count. A small confidence guard can bump uncertain low predictions upward so
runtime behavior stays conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_eng.intent.budgets import clamp
from context_eng.ml.features import features_to_vector
from context_eng.models import BudgetInfo

BUDGET_BUCKETS: tuple[int, ...] = (
    2000,
    3000,
    4000,
    5000,
    6000,
    8000,
    10000,
    12000,
    15000,
)


@dataclass(frozen=True)
class BudgetPrediction:
    """A classifier prediction after confidence handling and clamping."""

    budget: int
    raw_bucket: int
    confidence: float
    source: str = "random_forest_classifier"


def _require_joblib():
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "ML budget models require the 'ml' extra: pip install -e '.[ml]'"
        ) from exc
    return joblib


def _next_bucket(bucket: int) -> int:
    for candidate in BUDGET_BUCKETS:
        if candidate > bucket:
            return candidate
    return BUDGET_BUCKETS[-1]


class RandomForestBudgetModel:
    """Thin wrapper around a fitted sklearn RandomForestClassifier."""

    def __init__(
        self,
        classifier: Any,
        feature_names: list[str],
        confidence_threshold: float = 0.55,
    ):
        self.classifier = classifier
        self.feature_names = feature_names
        self.confidence_threshold = confidence_threshold

    def predict(
        self,
        features: dict[str, float | int],
        fixed_budget: BudgetInfo,
    ) -> BudgetPrediction:
        """Predict a safe budget bucket for ``features``.

        Low-confidence predictions below the fixed intent budget are bumped one
        bucket upward before final clamping.
        """
        values, names = features_to_vector(features)
        if names != self.feature_names:
            raise ValueError("feature vector does not match trained model columns")

        probabilities = self.classifier.predict_proba([values])[0]
        best_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
        raw_bucket = int(self.classifier.classes_[best_idx])
        confidence = float(probabilities[best_idx])

        budget = raw_bucket
        while confidence < self.confidence_threshold and budget < fixed_budget.recommended:
            next_budget = _next_bucket(budget)
            if next_budget == budget:
                break
            budget = next_budget

        query_tokens = int(features.get("query_tokens", 0))
        if query_tokens >= 35 and budget < 5000:
            budget = max(budget, 5000)
        if query_tokens >= 55 and budget < 8000:
            budget = max(budget, 8000)
        if query_tokens >= 75 and budget < 10000:
            budget = max(budget, 10000)

        budget = clamp(budget, BUDGET_BUCKETS[0], BUDGET_BUCKETS[-1])
        return BudgetPrediction(
            budget=budget,
            raw_bucket=raw_bucket,
            confidence=round(confidence, 4),
        )

    def save(self, path: str | Path) -> None:
        joblib = _require_joblib()
        payload = {
            "classifier": self.classifier,
            "feature_names": self.feature_names,
            "confidence_threshold": self.confidence_threshold,
            "budget_buckets": BUDGET_BUCKETS,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestBudgetModel":
        joblib = _require_joblib()
        payload = joblib.load(path)
        return cls(
            classifier=payload["classifier"],
            feature_names=list(payload["feature_names"]),
            confidence_threshold=float(payload.get("confidence_threshold", 0.55)),
        )
