"""Resolve and verify packaged RF budget model paths (H5).

Only models under ``context_eng/ml/models/`` with a pinned SHA-256 in
``checksums.json`` may be loaded. Arbitrary ``ml_model_path`` values are denied.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

# Runtime default: Random Forest trained on SWE-bench Lite oracle/BM25 labels.
SWEBENCH_MODEL_NAME = "budget_rf_swebench.joblib"
# Earlier fixture-query RF; still packaged, not loaded unless requested by name.
LEGACY_V2_MODEL_NAME = "budget_rf_v2.joblib"
DEFAULT_MODEL_NAME = SWEBENCH_MODEL_NAME

# Models ship beside this module: context_eng/ml/models/*.joblib
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_CHECKSUMS_PATH = _MODELS_DIR / "checksums.json"


def models_dir() -> Path:
    return _MODELS_DIR


def packaged_model_path(name: str = DEFAULT_MODEL_NAME) -> Path:
    """Return the filesystem path to a model shipped under ``context_eng.ml.models``."""
    path = _MODELS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Packaged RF model not found: {path} "
            "(expected under context_eng/ml/models/)"
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _pinned_checksums() -> dict[str, str]:
    if not _CHECKSUMS_PATH.is_file():
        return {}
    try:
        data = json.loads(_CHECKSUMS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v).lower() for k, v in data.items()}


def clear_checksum_cache() -> None:
    _pinned_checksums.cache_clear()


def verify_model_file(path: Path) -> Path:
    """Ensure ``path`` is a packaged model whose content matches the pin."""
    resolved = Path(path).resolve()
    models = _MODELS_DIR.resolve()
    if not resolved.is_relative_to(models):
        raise PermissionError(
            f"Refusing to load RF model outside packaged models dir: {resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"RF budget model not found: {resolved}")
    expected = _pinned_checksums().get(resolved.name)
    if not expected:
        raise PermissionError(
            f"No pinned checksum for model {resolved.name}; "
            f"update {_CHECKSUMS_PATH.name} after training"
        )
    actual = sha256_file(resolved)
    if actual != expected:
        raise PermissionError(
            f"Checksum mismatch for {resolved.name}: "
            f"expected {expected[:12]}… got {actual[:12]}…"
        )
    return resolved


def resolve_trusted_model_path(
    requested: Path | str | None = None,
    *,
    default_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    """Resolve a loadable model path, denying arbitrary filesystem paths."""
    if requested is None:
        return verify_model_file(packaged_model_path(default_name))
    path = Path(requested)
    if not path.is_absolute():
        # Allow bare filenames that refer to packaged models only.
        path = _MODELS_DIR / path.name
    return verify_model_file(path)
