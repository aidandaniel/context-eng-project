"""Unit tests for Grok bot PR-review helpers (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grok_pr_review.py"
_SPEC = importlib.util.spec_from_file_location("grok_pr_review", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
grok = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(grok)


def test_path_is_excluded_joblib_and_locks():
    assert grok.path_is_excluded("src/context_eng/ml/models/budget_rf_swebench.joblib")
    assert grok.path_is_excluded("package-lock.json")
    assert not grok.path_is_excluded("src/context_eng/server.py")


def test_truncate_diff_appends_notice():
    text = grok.truncate_diff("x" * 50, max_chars=20)
    assert text.startswith("x" * 20)
    assert "truncated" in text
    assert grok.truncate_diff("short", max_chars=20) == "short"


def test_filter_unified_diff_drops_joblib_keeps_py():
    diff = """diff --git a/src/context_eng/server.py b/src/context_eng/server.py
index 111..222 100644
--- a/src/context_eng/server.py
+++ b/src/context_eng/server.py
@@ -1 +1 @@
-old
+new
diff --git a/src/context_eng/ml/models/budget_rf_swebench.joblib b/src/context_eng/ml/models/budget_rf_swebench.joblib
index 111..222 100644
--- a/src/context_eng/ml/models/budget_rf_swebench.joblib
+++ b/src/context_eng/ml/models/budget_rf_swebench.joblib
@@ -1 +1 @@
-binary
+binary2
"""
    filtered = grok.filter_unified_diff(diff)
    assert "server.py" in filtered
    assert "+new" in filtered
    assert "budget_rf_swebench.joblib" not in filtered
