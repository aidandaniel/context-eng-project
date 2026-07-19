"""Train the Random Forest budget classifier from a labeled JSONL file.

Example:
  python scripts/train_budget_rf.py \\
    --data ml/data/swebench_lite_budget.jsonl \\
    --out ml/models/budget_rf_swebench.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from context_eng.ml.budget_model import RandomForestBudgetModel  # noqa: E402
from context_eng.ml.features import FEATURE_NAMES  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Labeled JSONL from build_swebench_budget_dataset.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "ml" / "models" / "budget_rf_swebench.joblib",
        help="Output joblib path",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction for a quick accuracy report (0 disables)",
    )
    args = parser.parse_args()

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit("scikit-learn is required to train the budget model") from exc

    rows = _load_jsonl(args.data)
    if not rows:
        raise SystemExit(f"No rows in {args.data}")

    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        features = row["features"]
        x.append([float(features[name]) for name in FEATURE_NAMES])
        y.append(int(row["label_budget"]))

    print(f"Loaded {len(rows)} rows from {args.data}")
    print("label_budget counts:", dict(sorted(Counter(y).items())))

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    if 0.0 < args.test_size < 1.0 and len(rows) >= 10:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y if len(set(y)) > 1 else None,
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        print(f"holdout accuracy: {accuracy_score(y_test, pred):.3f}")
        print(classification_report(y_test, pred, zero_division=0))
        # Refit on all data for the saved artifact.
        clf.fit(x, y)
    else:
        clf.fit(x, y)

    model = RandomForestBudgetModel(
        classifier=clf,
        feature_names=list(FEATURE_NAMES),
    )
    model.save(args.out)
    print(f"Wrote model -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
