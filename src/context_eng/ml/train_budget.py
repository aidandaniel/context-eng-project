"""Train a RandomForestClassifier for budget-bucket prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from context_eng.ml.budget_model import RandomForestBudgetModel
from context_eng.ml.features import FEATURE_NAMES, features_to_vector

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LABELS = _REPO_ROOT / "ml" / "data" / "budget_labels.jsonl"
_DEFAULT_OUTPUT = _REPO_ROOT / "ml" / "models" / "budget_rf_v2.joblib"


def _require_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the 'ml' extra: pip install -e '.[ml]'"
        ) from exc
    return RandomForestClassifier


def load_labels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train_model(
    rows: list[dict[str, Any]],
    *,
    n_estimators: int = 200,
    random_state: int = 17,
    confidence_threshold: float = 0.55,
) -> RandomForestBudgetModel:
    """Fit a Random Forest classifier from generated label rows."""
    RandomForestClassifier = _require_sklearn()
    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        values, names = features_to_vector(row["features"])
        if names != FEATURE_NAMES:
            raise ValueError("label row feature order does not match FEATURE_NAMES")
        x.append(values)
        y.append(int(row["y"]))

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
    )
    classifier.fit(x, y)
    return RandomForestBudgetModel(
        classifier=classifier,
        feature_names=list(FEATURE_NAMES),
        confidence_threshold=confidence_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v2 budget classifier.")
    parser.add_argument("--labels", type=Path, default=_DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    args = parser.parse_args()

    rows = load_labels(args.labels)
    model = train_model(
        rows,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        confidence_threshold=args.confidence_threshold,
    )
    model.save(args.output)
    print(f"Trained RandomForest budget model on {len(rows)} rows: {args.output}")


if __name__ == "__main__":
    main()
