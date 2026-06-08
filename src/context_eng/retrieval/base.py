"""Retriever protocol.

Defines the seam between candidate generation and the ranker so an
``EmbeddingRetriever`` can be dropped in later without changing tool schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from context_eng.models import CandidateChunk


@runtime_checkable
class Retriever(Protocol):
    def search(
        self, query: str, workspace: Path, limit: int
    ) -> list[CandidateChunk]:
        """Return up to ``limit`` candidate chunks relevant to ``query``."""
        ...
