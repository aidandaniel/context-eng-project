"""Packaged RF model path resolution."""

from __future__ import annotations

from context_eng.config import Config
from context_eng.ml.engine_budget import default_model_path
from context_eng.ml.model_paths import (
    DEFAULT_MODEL_NAME,
    SWEBENCH_MODEL_NAME,
    packaged_model_path,
)


def test_packaged_default_model_exists():
    path = packaged_model_path(DEFAULT_MODEL_NAME)
    assert path.is_file()
    assert path.name == DEFAULT_MODEL_NAME
    assert "context_eng" in path.parts
    assert "models" in path.parts


def test_packaged_swebench_model_exists():
    path = packaged_model_path(SWEBENCH_MODEL_NAME)
    assert path.is_file()


def test_default_model_path_uses_package_when_unset(tmp_path):
    cfg = Config(workspace_root=tmp_path)
    path = default_model_path(cfg)
    assert path == packaged_model_path(DEFAULT_MODEL_NAME)


def test_default_model_path_respects_override(tmp_path):
    custom = tmp_path / "custom.joblib"
    custom.write_bytes(b"x")
    cfg = Config(workspace_root=tmp_path, ml_model_path=custom)
    assert default_model_path(cfg) == custom
