"""Tests for quality eval helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.eval.quality import relevant_file_recall, task_rubric_pass
from context_eng.models import Chunk, ContextBundle, Intent

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks" / "fixture_repo"
TASK_EVAL = Path(__file__).resolve().parents[1] / "ml" / "data" / "task_eval_queries.yaml"


def _bundle(paths: list[str], content: str = "refreshToken logout verify_token") -> ContextBundle:
    chunks = [
        Chunk(
            path=p,
            start_line=1,
            end_line=3,
            content=content,
            score=1.0,
            reason="test",
            tokens=10,
        )
        for p in paths
    ]
    return ContextBundle(
        intent=Intent.DEBUG,
        budget_used=10,
        budget_limit=8000,
        chunks=chunks,
        excluded_summary="",
        bundle_id="test-bundle",
        optional_chunks_used=0,
    )


def test_relevant_file_recall_all_present():
    recall = relevant_file_recall(
        ["src/auth/refresh.py", "src/auth/tokens.py"],
        ["src/auth/refresh.py", "src/auth/tokens.py", "src/api/routes.py"],
    )
    assert recall == 1.0


def test_relevant_file_recall_partial():
    recall = relevant_file_recall(
        ["src/auth/refresh.py", "src/auth/tokens.py"],
        ["src/auth/refresh.py"],
    )
    assert recall == 0.5


def test_task_rubric_pass_expected_paths_and_content():
    bundle = _bundle(["src/auth/refresh.py"], content="refreshToken rotation")
    rubric = {
        "expected_paths": ["src/auth/refresh.py"],
        "content_contains": ["refreshToken"],
    }
    assert task_rubric_pass(bundle, rubric)


def test_task_rubric_pass_any_of_paths():
    bundle = _bundle(["src/users/models.py"])
    rubric = {
        "any_of_paths": [
            ["src/users/service.py"],
            ["src/users/models.py"],
        ],
    }
    assert task_rubric_pass(bundle, rubric)


def test_engine_bundle_passes_fixture_task_eval_queries():
    rows = yaml.safe_load(TASK_EVAL.read_text(encoding="utf-8"))
    engine = ContextEngine(config=Config(workspace_root=FIXTURE))
    recalls: list[float] = []
    rubric_passes = 0
    for row in rows:
        bundle = engine.get_context_bundle(row["query"])
        paths = [c.path for c in bundle.chunks]
        recalls.append(relevant_file_recall(row["relevant_files"], paths))
        if task_rubric_pass(bundle, row.get("rubric") or {}):
            rubric_passes += 1
    assert sum(recalls) / len(recalls) >= 0.5
    assert rubric_passes >= len(rows) // 2
