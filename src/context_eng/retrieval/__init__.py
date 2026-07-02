"""Retrieval layer: candidate generation from the workspace."""

from context_eng.retrieval.base import Retriever
from context_eng.retrieval.composite_retriever import CompositeRetriever, build_retriever
from context_eng.retrieval.embedding_retriever import EmbeddingRetriever
from context_eng.retrieval.grep_retriever import GrepRetriever

__all__ = [
    "Retriever",
    "GrepRetriever",
    "EmbeddingRetriever",
    "CompositeRetriever",
    "build_retriever",
]
