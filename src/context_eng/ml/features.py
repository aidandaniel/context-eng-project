"""ML feature extraction for budget prediction (Step 1)."""

from __future__ import annotations

import math
from pathlib import Path

from context_eng.config import Config
from context_eng.models import Intent, QueryAnalysis
from context_eng.workspace import iter_files, read_text

INTENT_COLUMNS = [
    "intent_debug",
    "intent_implement",
    "intent_explain",
    "intent_refactor",
    "intent_review",
]

BASE_FEATURE_NAMES = [
    "query_tokens",
    "mentioned_files",
    "mentioned_symbols",
    "has_stack_trace",
    "has_error_token",
    "intent_confidence",
    "repo_file_count",
    "repo_loc_log",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + INTENT_COLUMNS


def repo_stats(config: Config) -> tuple[int, float]:
    """Walk workspace (respecting ignore_globs), return (file_count, log10(loc+1))."""
    workspace = Path(config.workspace_root).resolve()
    file_count = 0
    total_lines = 0
    for path in iter_files(workspace, config.ignore_globs):
        file_count += 1
        try:
            text = read_text(path)
        except OSError:
            continue
        total_lines += text.count("\n") + (1 if text else 0)
    return file_count, math.log10(total_lines + 1)


def _intent_one_hot(intent: Intent) -> dict[str, int]:
    active = f"intent_{intent.value}"
    return {col: (1 if col == active else 0) for col in INTENT_COLUMNS}


def extract_features(
    query: str,
    analysis: QueryAnalysis,
    config: Config,
) -> dict[str, float | int]:
    """Flat feature dict for ML. Values come from ``analysis`` and ``config``."""
    _ = query  # API symmetry for label gen; v1 reads analysis only
    signals = analysis.signals
    file_count, loc_log = repo_stats(config)
    features: dict[str, float | int] = {
        "query_tokens": signals.query_tokens,
        "mentioned_files": len(signals.mentioned_files),
        "mentioned_symbols": len(signals.mentioned_symbols),
        "has_stack_trace": int(signals.has_stack_trace),
        "has_error_token": int(signals.has_error_token),
        "intent_confidence": analysis.confidence,
        "repo_file_count": file_count,
        "repo_loc_log": loc_log,
    }
    features.update(_intent_one_hot(analysis.intent))
    return features


def features_to_vector(
    features: dict[str, float | int],
) -> tuple[list[float], list[str]]:
    """Convert dict to ordered numeric vector for sklearn."""
    values = [float(features[name]) for name in FEATURE_NAMES]
    return values, list(FEATURE_NAMES)
