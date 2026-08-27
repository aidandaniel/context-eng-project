"""Broken context-eng.toml must surface config_error instead of silent defaults."""

from __future__ import annotations

from pathlib import Path

from context_eng.config import load_config


def test_invalid_toml_sets_config_error(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONTEXT_ENG_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("CONTEXT_ENG_WORKSPACE", raising=False)
    (tmp_path / "context-eng.toml").write_text("[[[not valid", encoding="utf-8")
    cfg = load_config(str(tmp_path))
    assert cfg.config_error is not None
    assert "Invalid context-eng.toml" in cfg.config_error
    assert cfg.default_max_tokens == 8000
