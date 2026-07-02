"""Composite retriever: grep primary, optional embedding merge."""

from __future__ import annotations

from pathlib import Path

from context_eng.config import Config
from context_eng.models import CandidateChunk
from context_eng.retrieval.base import Retriever
from context_eng.retrieval.grep_retriever import GrepRetriever


def _chunk_key(chunk: CandidateChunk) -> tuple[str, int, int]:
    return (chunk.path, chunk.start_line, chunk.end_line)


def merge_retrieval_hits(
    primary: list[CandidateChunk],
    secondary: list[CandidateChunk],
    limit: int,
) -> list[CandidateChunk]:
    """Merge ``secondary`` into ``primary``, deduping by path/line span."""
    seen = {_chunk_key(c) for c in primary}
    merged = list(primary)
    for chunk in secondary:
        key = _chunk_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged[:limit]


class CompositeRetriever:
    """Grep-first retriever with optional local embedding augmentation."""

    def __init__(self, config: Config):
        self.config = config
        self._grep = GrepRetriever(config)
        self._embedding: Retriever | None = None
        if config.enable_embedding_retriever:
            from context_eng.retrieval.embedding_retriever import EmbeddingRetriever

            self._embedding = EmbeddingRetriever(config)

    def search(
        self, query: str, workspace: Path, limit: int
    ) -> list[CandidateChunk]:
        grep_hits = self._grep.search(query, workspace, limit)
        if self._embedding is None:
            return grep_hits
        embed_hits = self._embedding.search(query, workspace, limit)
        return merge_retrieval_hits(grep_hits, embed_hits, limit)


def build_retriever(config: Config) -> Retriever:
    """Factory used by the engine; embeddings off == grep-only behavior."""
    return CompositeRetriever(config)
