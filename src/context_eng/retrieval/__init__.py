"""Retrieval layer: candidate generation from the workspace."""

from context_eng.retrieval.base import Retriever
from context_eng.retrieval.grep_retriever import GrepRetriever

__all__ = ["Retriever", "GrepRetriever"]
