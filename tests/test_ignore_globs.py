"""H3: real ignore globs + merge overrides; secrets stay separate."""

from __future__ import annotations

from pathlib import Path

from context_eng.config import DEFAULT_IGNORE_GLOBS, Config, load_config
from context_eng.ignore import merge_ignore_globs, path_is_ignored
from context_eng.workspace import is_secret_path, iter_files


def test_path_is_ignored_supports_star_globs():
    assert path_is_ignored("certs/server.pem", ("*.pem",))
    assert path_is_ignored("vendor/pkg/lib.py", ("vendor",))
    assert path_is_ignored("a/node_modules/x.js", ("node_modules",))
    assert not path_is_ignored("src/app.py", ("*.pem", "node_modules"))


def test_merge_ignore_globs_preserves_defaults():
    merged = merge_ignore_globs(DEFAULT_IGNORE_GLOBS, ("*.pem", "vendor"))
    assert ".git" in merged
    assert ".venv" in merged
    assert "*.pem" in merged
    assert "vendor" in merged


def test_toml_ignore_globs_merge_not_replace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CONTEXT_ENG_ALLOWED_ROOTS", str(tmp_path))
    (tmp_path / "context-eng.toml").write_text(
        '[context_eng]\nignore_globs = ["*.pem", "vendor"]\n',
        encoding="utf-8",
    )
    cfg = load_config(str(tmp_path))
    assert ".git" in cfg.ignore_globs
    assert ".venv" in cfg.ignore_globs
    assert "*.pem" in cfg.ignore_globs
    assert "vendor" in cfg.ignore_globs


def test_iter_files_honors_pem_glob_and_secret_denylist(tmp_path: Path):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "leak.pem").write_text("SECRET\n", encoding="utf-8")
    (tmp_path / ".env").write_text("KEY=1\n", encoding="utf-8")
    cfg = Config(workspace_root=tmp_path, ignore_globs=(".git", "*.pem"))
    found = {p.name for p in iter_files(tmp_path, cfg.ignore_globs)}
    assert "ok.py" in found
    assert "leak.pem" not in found
    assert ".env" not in found
    assert is_secret_path(tmp_path / ".env")
