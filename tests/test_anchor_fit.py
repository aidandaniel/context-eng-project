"""Tests for anchor budget auto-fit."""

import textwrap
from pathlib import Path

from context_eng.anchors.fit import ensure_budget_fits_anchors, estimate_must_include_tokens
from context_eng.ml.budget_model import BUDGET_BUCKETS


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_estimate_must_include_tokens_uses_head_lines(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "src" / "a.py",
        """
        # line
        x = 1
        """,
    )
    tokens = estimate_must_include_tokens(["src/a.py"], tmp_path)
    assert tokens > 0


def test_estimate_must_include_tokens_sums_multiple_anchors(tmp_path: Path) -> None:
    for name in ("a.py", "b.py"):
        _write_file(tmp_path / "src" / name, "value = 42\n" * 30)
    single = estimate_must_include_tokens(["src/a.py"], tmp_path)
    both = estimate_must_include_tokens(["src/a.py", "src/b.py"], tmp_path)
    assert both > single


def test_ensure_budget_fits_anchors_no_change_when_within_ceiling(tmp_path: Path) -> None:
    _write_file(tmp_path / "tiny.py", "x = 1\n")
    fitted = ensure_budget_fits_anchors(8000, ["tiny.py"], tmp_path)
    assert fitted == 8000


def test_ensure_budget_fits_anchors_bumps_to_next_bucket(tmp_path: Path) -> None:
    # Four heavy head slices (~8k tokens) exceed 1.5x the 4000 bucket.
    for i in range(4):
        _write_file(
            tmp_path / "src" / f"big{i}.py",
            ("word " * 50 + "\n") * 40,
        )
    paths = [f"src/big{i}.py" for i in range(4)]
    estimated = estimate_must_include_tokens(paths, tmp_path)
    assert estimated > int(4000 * 1.5)

    fitted = ensure_budget_fits_anchors(4000, paths, tmp_path)
    assert fitted > 4000
    assert fitted in BUDGET_BUCKETS
    assert int(fitted * 1.5) >= estimated


def test_ensure_budget_fits_anchors_empty_paths() -> None:
    assert ensure_budget_fits_anchors(5000, [], Path(".")) == 5000
