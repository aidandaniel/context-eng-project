"""Tier-1 RF training/eval dashboard (matplotlib)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from context_eng.ml.budget_model import BUDGET_BUCKETS, RandomForestBudgetModel
from context_eng.ml.features import FEATURE_NAMES


def _require_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "context-eng-matplotlib")
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RF dashboard requires matplotlib (pip install -e '.[ml]')"
        ) from exc
    return plt, confusion_matrix, precision_recall_fscore_support


def _bucket_labels(buckets: list[int]) -> list[str]:
    return [f"{b // 1000}k" if b >= 1000 else str(b) for b in buckets]


def _label_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        bucket = int(row["y"])
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _feature_importance(model_path: Path, top_n: int) -> list[tuple[str, float]]:
    if not model_path.is_file():
        return []
    model = RandomForestBudgetModel.load(model_path)
    importances = list(model.classifier.feature_importances_)
    names = list(model.feature_names)
    if len(names) != len(importances):
        names = list(FEATURE_NAMES)[: len(importances)]
    ranked = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def write_rf_dashboard(
    *,
    rows: list[dict[str, Any]],
    y_true: list[int],
    y_pred: list[int],
    model_path: Path,
    output_path: Path,
    top_features: int = 15,
) -> Path:
    """Write a 2x2 PNG: label dist, confusion matrix, feature importance, P/R per bucket."""
    plt, confusion_matrix, precision_recall_fscore_support = _require_matplotlib()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RF budget model — training & CV summary", fontsize=14, y=0.98)

    # --- 1. Label distribution ---
    ax_labels = axes[0, 0]
    counts = _label_counts(rows)
    buckets_present = sorted(counts.keys())
    if buckets_present:
        values = [counts[b] for b in buckets_present]
        ax_labels.bar(
            _bucket_labels(buckets_present),
            values,
            color="#4c78a8",
            edgecolor="white",
        )
        ax_labels.set_title("Label distribution (training y)")
        ax_labels.set_xlabel("Budget bucket")
        ax_labels.set_ylabel("Count")
        ax_labels.tick_params(axis="x", rotation=45)
    else:
        ax_labels.text(0.5, 0.5, "No labels", ha="center", va="center")
        ax_labels.set_axis_off()

    # --- 2. Confusion matrix (CV predictions) ---
    ax_cm = axes[0, 1]
    if y_true and y_pred and len(y_true) == len(y_pred):
        label_set = sorted(set(y_true) | set(y_pred))
        if len(label_set) == 1:
            only = label_set[0]
            ax_cm.text(
                0.5,
                0.55,
                f"All labels & CV preds:\n{only} ({only // 1000}k)",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax_cm.text(
                0.5,
                0.35,
                f"n={len(y_true)} (single bucket)",
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
            )
            ax_cm.set_title("Confusion matrix (stratified CV)")
            ax_cm.set_axis_off()
        else:
            cm = confusion_matrix(y_true, y_pred, labels=label_set)
            im = ax_cm.imshow(cm, cmap="Blues")
            ax_cm.set_xticks(range(len(label_set)), _bucket_labels(label_set), rotation=45)
            ax_cm.set_yticks(range(len(label_set)), _bucket_labels(label_set))
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            ax_cm.set_title("Confusion matrix (stratified CV)")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
            fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    else:
        ax_cm.text(0.5, 0.5, "CV predictions unavailable", ha="center", va="center")
        ax_cm.set_axis_off()

    # --- 3. Feature importance (fitted production model) ---
    ax_fi = axes[1, 0]
    ranked = _feature_importance(model_path, top_features)
    if ranked:
        names = [name.replace("intent_", "i.") for name, _ in reversed(ranked)]
        fi_values: list[float] = [float(val) for _, val in reversed(ranked)]
        ax_fi.barh(names, fi_values, color="#72b7b2", edgecolor="white")
        ax_fi.set_title(f"Feature importance (top {len(ranked)}, Gini)")
        ax_fi.set_xlabel("Importance")
    else:
        ax_fi.text(
            0.5,
            0.5,
            f"Model not found:\n{model_path.name}",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax_fi.set_axis_off()

    # --- 4. Per-bucket precision / recall (CV) ---
    ax_pr = axes[1, 1]
    if y_true and y_pred and len(y_true) == len(y_pred):
        label_set = sorted(set(y_true) | set(y_pred))
        if len(label_set) == 1:
            ax_pr.text(
                0.5,
                0.5,
                "Precision/recall N/A\n(single bucket)",
                ha="center",
                va="center",
            )
            ax_pr.set_title("Per-bucket precision & recall (CV)")
            ax_pr.set_axis_off()
        else:
            precision, recall, _, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=label_set,
                zero_division=0,
            )
            x_pos = range(len(label_set))
            width = 0.35
            ax_pr.bar(
                [p - width / 2 for p in x_pos],
                precision,
                width,
                label="Precision",
                color="#f58518",
            )
            ax_pr.bar(
                [p + width / 2 for p in x_pos],
                recall,
                width,
                label="Recall",
                color="#54a24b",
            )
            ax_pr.set_xticks(list(x_pos), _bucket_labels(label_set), rotation=45)
            ax_pr.set_ylim(0, 1.05)
            ax_pr.set_title("Per-bucket precision & recall (CV)")
            ax_pr.set_ylabel("Score")
            ax_pr.legend(loc="lower right", fontsize=8)
            for idx, sup in enumerate(support):
                if sup:
                    ax_pr.text(
                        idx,
                        1.02,
                        f"n={sup}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
    else:
        ax_pr.text(0.5, 0.5, "CV metrics unavailable", ha="center", va="center")
        ax_pr.set_axis_off()

    # Reference: full bucket ladder (footer)
    ladder = ", ".join(_bucket_labels(list(BUDGET_BUCKETS)))
    fig.text(0.5, 0.01, f"Buckets: {ladder}", ha="center", fontsize=8, color="#666666")

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
