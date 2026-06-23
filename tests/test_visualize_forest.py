"""Tests for Random Forest visualization helpers."""

import pytest

from context_eng.ml.features import FEATURE_NAMES
from context_eng.ml.visualize_forest import labels_to_xy, select_tree


def _row(label: int):
    return {
        "features": {name: 0 for name in FEATURE_NAMES},
        "y": label,
    }


def test_labels_to_xy_uses_feature_order():
    x, y = labels_to_xy([_row(2000), _row(4000)])

    assert len(x) == 2
    assert len(x[0]) == len(FEATURE_NAMES)
    assert y.tolist() == [2000, 4000]


class _FakeForest:
    estimators_ = ["tree-0", "tree-1"]


def test_select_tree_by_index():
    assert select_tree(_FakeForest(), 1) == "tree-1"


def test_select_tree_rejects_invalid_index():
    with pytest.raises(IndexError):
        select_tree(_FakeForest(), 2)
