"""Permit pytest tmp directories in the workspace jail (cwd remains allowed)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _allow_pytest_tmp_and_cwd(tmp_path_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = Path.cwd().resolve()
    try:
        base = tmp_path_factory.getbasetemp().resolve()
    except (OSError, PermissionError):
        base = cwd
    roots = os.pathsep.join([str(cwd), str(base), str(base.parent)])
    monkeypatch.setenv("CONTEXT_ENG_ALLOWED_ROOTS", roots)
