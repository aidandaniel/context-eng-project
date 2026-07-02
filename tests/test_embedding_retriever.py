"""Tests for composite and embedding retrievers."""

from __future__ import annotations

import textwrap

from pathlib import Path

import pytest

from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.retrieval.composite_retriever import (
    CompositeRetriever,
    build_retriever,
    merge_retrieval_hits,
)
from context_eng.retrieval.embedding_retriever import EmbeddingRetriever
from context_eng.retrieval.grep_retriever import GrepRetriever
from context_eng.models import CandidateChunk

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks" / "fixture_repo"


def _chunk(path: str, start: int = 1, end: int = 5, tier: str = "grep") -> CandidateChunk:
    return CandidateChunk(
        path=path,
        start_line=start,
        end_line=end,
        content="sample content",
        tier=tier,
    )


def test_merge_retrieval_hits_dedupes_and_respects_limit():
    primary = [_chunk("a.py"), _chunk("b.py")]
    secondary = [_chunk("b.py"), _chunk("c.py", tier="embedding")]
    merged = merge_retrieval_hits(primary, secondary, limit=3)
    paths = [c.path for c in merged]
    assert paths == ["a.py", "b.py", "c.py"]


def test_composite_with_embeddings_off_matches_grep(tmp_path):
    (tmp_path / "alpha.py").write_text(
        textwrap.dedent(
            """
            def alpha_handler():
                return "alpha"
            """
        ).strip(),
        encoding="utf-8",
    )
    (tmp_path / "beta.py").write_text("def beta_handler():\n    return 1\n", encoding="utf-8")

    config = Config(workspace_root=tmp_path, enable_embedding_retriever=False)
    query = "Where is alpha_handler defined?"
    limit = 10
    grep_hits = GrepRetriever(config).search(query, tmp_path, limit)
    composite_hits = CompositeRetriever(config).search(query, tmp_path, limit)
    assert [(c.path, c.start_line, c.end_line) for c in grep_hits] == [
        (c.path, c.start_line, c.end_line) for c in composite_hits
    ]


def test_build_retriever_default_is_grep_only():
    config = Config(workspace_root=FIXTURE, enable_embedding_retriever=False)
    retriever = build_retriever(config)
    assert isinstance(retriever, CompositeRetriever)
    hits = retriever.search("refreshToken logout", FIXTURE, 20)
    assert hits


def test_engine_uses_composite_retriever_with_embeddings_off():
    config = Config(workspace_root=FIXTURE, enable_embedding_retriever=False)
    engine = ContextEngine(config=config)
    bundle = engine.get_context_bundle("Why does refreshToken fail immediately after logout?")
    assert bundle.budget_limit > 0
    assert len(bundle.chunks) > 0


def test_embedding_retriever_graceful_without_optional_dep(tmp_path):
    (tmp_path / "service.py").write_text("def renew_session():\n    pass\n", encoding="utf-8")
    config = Config(workspace_root=tmp_path, enable_embedding_retriever=True)
    retriever = EmbeddingRetriever(config)
    hits = retriever.search("renew session after logout", tmp_path, 5)
    assert hits == []


def test_embedding_retriever_returns_hits_when_model_cached(tmp_path):
    pytest.importorskip("sentence_transformers")
    retriever = EmbeddingRetriever(
        Config(workspace_root=tmp_path, enable_embedding_retriever=True)
    )
    if retriever._load_model() is None:
        pytest.skip("all-MiniLM-L6-v2 not in local cache")

    (tmp_path / "payments.py").write_text(
        textwrap.dedent(
            """
            def process_card_charge(amount, gateway):
                return gateway.charge(amount)
            """
        ).strip(),
        encoding="utf-8",
    )
    hits = retriever.search("card charge payment gateway", tmp_path, 5)
    assert any(c.path == "payments.py" for c in hits)
