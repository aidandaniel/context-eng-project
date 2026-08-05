"""Resolve packaged RF budget model paths."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_NAME = "budget_rf_v2.joblib"
SWEBENCH_MODEL_NAME = "budget_rf_swebench.joblib"

# Models ship beside this module: context_eng/ml/models/*.joblib
_MODELS_DIR = Path(__file__).resolve().parent / "models"


def packaged_model_path(name: str = DEFAULT_MODEL_NAME) -> Path:
    """Return the filesystem path to a model shipped under ``context_eng.ml.models``."""
    path = _MODELS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Packaged RF model not found: {path} "
            "(expected under context_eng/ml/models/)"
        )
    return path
