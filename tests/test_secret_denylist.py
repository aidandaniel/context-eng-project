"""Secret-prone paths must not be retrieval-eligible via iter_files."""

from __future__ import annotations

from pathlib import Path

from context_eng.config import DEFAULT_IGNORE_GLOBS
from context_eng.workspace import is_secret_path, iter_files


def test_is_secret_path_basenames():
    assert is_secret_path(Path(".env"))
    assert is_secret_path(Path(".env.local"))
    assert is_secret_path(Path("credentials.json"))
    assert is_secret_path(Path("foo.pem"))
    assert is_secret_path(Path("id_rsa"))
    assert is_secret_path(Path(".npmrc"))
    assert not is_secret_path(Path("main.py"))
    assert not is_secret_path(Path("config.json"))


def test_iter_files_skips_secrets_keeps_py(tmp_path: Path):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (tmp_path / "credentials.json").write_text('{"k": "v"}\n', encoding="utf-8")
    (tmp_path / "foo.pem").write_text("-----BEGIN-----\n", encoding="utf-8")

    found = {p.name for p in iter_files(tmp_path, DEFAULT_IGNORE_GLOBS)}
    assert "main.py" in found
    assert ".env" not in found
    assert "credentials.json" not in found
    assert "foo.pem" not in found
