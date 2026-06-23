"""Visualize a trained RandomForestClassifier tree with dtreeviz."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from context_eng.ml.budget_model import RandomForestBudgetModel
from context_eng.ml.features import FEATURE_NAMES, features_to_vector
from context_eng.ml.train_budget import load_labels

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LABELS = _REPO_ROOT / "ml" / "data" / "budget_labels.jsonl"
_DEFAULT_MODEL = _REPO_ROOT / "ml" / "models" / "budget_rf_v2.joblib"
_DEFAULT_OUTPUT = _REPO_ROOT / "ml" / "reports" / "budget_rf_tree.svg"
_COMMON_GRAPHVIZ_BINS = (
    Path(r"C:\Program Files\Graphviz\bin"),
    Path(r"C:\Program Files (x86)\Graphviz\bin"),
)


def _require_dtreeviz():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "context-eng-matplotlib")
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from dtreeviz import model
    except ImportError as exc:
        raise RuntimeError(
            "Tree visualization requires the 'viz' extra: pip install -e '.[viz]'"
        ) from exc
    return model


def labels_to_xy(rows: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Convert generated label rows into sklearn-compatible X/y arrays."""
    import numpy as np

    x: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        values, names = features_to_vector(row["features"])
        if names != FEATURE_NAMES:
            raise ValueError("label row feature order does not match FEATURE_NAMES")
        x.append(values)
        y.append(int(row["y"]))
    return np.array(x), np.array(y)


def select_tree(classifier: Any, tree_index: int) -> Any:
    """Return one estimator from a fitted RandomForestClassifier."""
    estimators = getattr(classifier, "estimators_", None)
    if not estimators:
        raise ValueError("model does not contain fitted forest estimators")
    if tree_index < 0 or tree_index >= len(estimators):
        raise IndexError(
            f"tree_index {tree_index} outside forest size {len(estimators)}"
        )
    return estimators[tree_index]


def ensure_graphviz_dot() -> None:
    """Make the native Graphviz dot executable discoverable when possible."""
    if shutil.which("dot") is not None:
        return
    for bin_dir in _COMMON_GRAPHVIZ_BINS:
        if (bin_dir / "dot.exe").is_file():
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            return
    raise RuntimeError(
        "dtreeviz requires the Graphviz 'dot' executable on PATH. "
        "Install Graphviz from https://graphviz.org/download/ and retry."
    )


def render_tree(
    *,
    model_path: Path,
    labels_path: Path,
    output_path: Path,
    tree_index: int,
) -> Path:
    """Render one tree from a saved Random Forest model to SVG."""
    ensure_graphviz_dot()
    dtreeviz_model = _require_dtreeviz()
    budget_model = RandomForestBudgetModel.load(model_path)
    rows = load_labels(labels_path)
    x, y = labels_to_xy(rows)
    tree = select_tree(budget_model.classifier, tree_index)
    class_names = [str(cls) for cls in budget_model.classifier.classes_]

    viz_model = dtreeviz_model(
        tree,
        X_train=x,
        y_train=y,
        feature_names=list(FEATURE_NAMES),
        target_name="budget_bucket",
        class_names=class_names,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    viz_model.view().save(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one tree from the trained budget Random Forest."
    )
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=_DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--tree-index", type=int, default=0)
    args = parser.parse_args()

    output = render_tree(
        model_path=args.model,
        labels_path=args.labels,
        output_path=args.output,
        tree_index=args.tree_index,
    )
    print(f"Wrote dtreeviz tree visualization to {output}")


if __name__ == "__main__":
    main()
