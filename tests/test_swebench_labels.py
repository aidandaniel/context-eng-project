"""Unit tests for SWE-bench Lite budget labeling (no HF download)."""

from __future__ import annotations

from pathlib import Path

from context_eng.config import Config
from context_eng.ml.budget_model import snap_to_bucket
from context_eng.ml.features import FEATURE_NAMES
from context_eng.ml.swebench_labels import (
    build_example,
    build_examples_from_splits,
    context_token_count,
    extract_code_block,
    files_in_code_block,
    files_in_patch,
    label_budget_from_oracle_tokens,
)


_ORACLE_TEXT = """You will be provided with a partial code base and an issue statement explaining a problem to resolve.
<issue>
Fix bug in `separability_matrix` for nested models
</issue>
<code>
[start of astropy/modeling/separable.py]
def _cstack(left, right):
    return left
[end of astropy/modeling/separable.py]
[start of README.rst]
Astropy
[end of README.rst]
</code>
"""

_BM13_TEXT = """You will be provided with a partial code base and an issue statement explaining a problem to resolve.
<issue>
Fix bug in `separability_matrix` for nested models
</issue>
<code>
[start of a.py]
""" + ("x = 1\n" * 400) + """
[end of a.py]
[start of b.py]
""" + ("y = 2\n" * 400) + """
[end of b.py]
</code>
"""

_BM27_TEXT = """You will be provided with a partial code base and an issue statement explaining a problem to resolve.
<issue>
Fix bug in `separability_matrix` for nested models
</issue>
<code>
[start of a.py]
""" + ("x = 1\n" * 800) + """
[end of a.py]
[start of b.py]
""" + ("y = 2\n" * 800) + """
[end of b.py]
[start of c.py]
""" + ("z = 3\n" * 800) + """
[end of c.py]
</code>
"""

_PATCH = """diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -1 +1 @@
-old
+new
"""


def test_extract_code_and_files():
    code = extract_code_block(_ORACLE_TEXT)
    assert "separable.py" in code
    assert files_in_code_block(code) == [
        "astropy/modeling/separable.py",
        "README.rst",
    ]
    assert files_in_patch(_PATCH) == ["astropy/modeling/separable.py"]


def test_label_snaps_to_bucket():
    tokens = context_token_count(_ORACLE_TEXT)
    assert tokens > 0
    assert label_budget_from_oracle_tokens(tokens) == snap_to_bucket(tokens)
    assert label_budget_from_oracle_tokens(0) == 2000


def test_build_example_feature_schema():
    example = build_example(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        problem_statement="Fix bug and error in separability_matrix traceback",
        patch=_PATCH,
        oracle_text=_ORACLE_TEXT,
        bm25_13k_text=_BM13_TEXT,
        bm25_27k_text=_BM27_TEXT,
        config=Config(workspace_root=Path(".").resolve()),
    )
    assert example.label_source == "swebench_lite_oracle_tokens"
    assert example.oracle_tokens > 0
    assert example.bm25_13k_tokens > example.oracle_tokens
    assert example.bm25_27k_tokens > example.bm25_13k_tokens
    assert set(example.features) == set(FEATURE_NAMES)
    assert example.features["discovered_anchor_count"] == 2
    assert example.features["must_include_token_estimate"] != example.oracle_tokens
    assert example.features["must_include_token_estimate"] >= 2000
    assert example.features["intent_debug"] == 1


def test_build_examples_from_splits_joins_and_limits():
    lite = [
        {
            "instance_id": "a__1",
            "repo": "a/a",
            "problem_statement": "fix crash error",
            "patch": _PATCH,
        },
        {
            "instance_id": "b__2",
            "repo": "b/b",
            "problem_statement": "fix crash error",
            "patch": _PATCH,
        },
    ]
    oracle = [
        {"instance_id": "a__1", "text": _ORACLE_TEXT, "repo": "a/a"},
        {"instance_id": "b__2", "text": _ORACLE_TEXT, "repo": "b/b"},
    ]
    bm13 = [
        {"instance_id": "a__1", "text": _BM13_TEXT},
        {"instance_id": "b__2", "text": _BM13_TEXT},
    ]
    bm27 = [
        {"instance_id": "a__1", "text": _BM27_TEXT},
        {"instance_id": "b__2", "text": _BM27_TEXT},
    ]
    examples = build_examples_from_splits(
        lite,
        oracle,
        bm13,
        bm27,
        config=Config(workspace_root=Path(".").resolve()),
        limit=1,
    )
    assert len(examples) == 1
    assert examples[0].instance_id == "a__1"
