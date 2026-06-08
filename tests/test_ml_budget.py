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
from context_eng.models import Intent

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks" / "fixture_repo"


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
