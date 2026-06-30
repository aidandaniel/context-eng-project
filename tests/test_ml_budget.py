"""Tests for ML budget feature extraction (Step 1)."""

from pathlib import Path

from context_eng.config import Config
from context_eng.intent.classifier import analyze
from context_eng.ml.features import (
    FEATURE_NAMES,
    INTENT_COLUMNS,
    extract_features,
    features_to_vector,
)
from context_eng.ml.generate_labels import label_all
from context_eng.ml.budget_model import BUDGET_BUCKETS, RandomForestBudgetModel
from context_eng.models import BudgetInfo
from context_eng.models import Intent

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks" / "fixture_repo"
TRAINING_QUERIES = (
    Path(__file__).resolve().parents[1] / "ml" / "data" / "budget_training_queries.yaml"
)


def test_feature_extraction():
    cfg = Config(workspace_root=FIXTURE)
    query = "Fix TypeError in src/auth/refresh.py"
    analysis = analyze(query, cfg)
    feats = extract_features(query, analysis, cfg)
    vec, names = features_to_vector(feats)

    assert names == FEATURE_NAMES
    assert len(vec) == len(FEATURE_NAMES)
    assert feats["query_tokens"] > 0
    assert feats["mentioned_files"] >= 1
    assert feats["repo_file_count"] > 20
    assert 3.0 <= feats["repo_loc_log"] <= 5.0
    intent_cols = [feats[c] for c in INTENT_COLUMNS]
    assert sum(intent_cols) == 1
    assert analysis.intent == Intent.DEBUG
    assert feats["intent_debug"] == 1


class _FakeClassifier:
    classes_ = [2000, 4000]

    def predict_proba(self, rows):
        assert len(rows) == 1
        return [[0.51, 0.49]]


def test_random_forest_budget_model_bumps_uncertain_low_prediction():
    features = {name: 0 for name in FEATURE_NAMES}
    features["query_tokens"] = 10
    model = RandomForestBudgetModel(
        classifier=_FakeClassifier(),
        feature_names=list(FEATURE_NAMES),
        confidence_threshold=0.55,
    )
    fixed = BudgetInfo(recommended=4000, min=2000, max=6000)

    prediction = model.predict(features, fixed)

    assert prediction.raw_bucket == 2000
    assert prediction.confidence == 0.51
    assert prediction.budget == 4000


def test_training_corpus_reaches_multiple_budget_buckets():
    rows = label_all(FIXTURE, TRAINING_QUERIES)
    budgets = {row["y"] for row in rows}
    high_budget_rows = [row for row in rows if row["y"] > 2000]
    upper_buckets = {bucket for bucket in BUDGET_BUCKETS if bucket >= 5000}

    assert any(budget > 2000 for budget in budgets)
    assert len(upper_buckets & budgets) >= 4
    assert len(high_budget_rows) >= 12
    assert max(budgets) >= 10000
    assert len(budgets) >= 3
    for row in rows:
        assert row["expected_tokens"] > 0
