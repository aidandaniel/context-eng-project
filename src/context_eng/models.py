"""Pydantic models shared across the Context Engineering MCP server."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Coarse task intent inferred from the user query."""

    DEBUG = "debug"
    IMPLEMENT = "implement"
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    REVIEW = "review"


class QuerySignals(BaseModel):
    """Cheap features extracted from a query, reused for logging/ML later."""

    has_stack_trace: bool = False
    mentioned_files: list[str] = Field(default_factory=list)
    inferred_files: list[str] = Field(default_factory=list)
    mentioned_symbols: list[str] = Field(default_factory=list)
    has_error_token: bool = False
    query_tokens: int = 0


class BudgetInfo(BaseModel):
    """Recommended token budget for a query, with clamp bounds."""

    recommended: int
    min: int
    max: int
    source: str = "fixed_intent_table"


class QueryAnalysis(BaseModel):
    """Result of `analyze_query`."""

    intent: Intent
    confidence: float
    signals: QuerySignals
    budget: BudgetInfo


class CandidateChunk(BaseModel):
    """A retrieval candidate before ranking/packing."""

    path: str
    start_line: int
    end_line: int
    content: str
    tier: str = "grep"
    keyword_match: float = 0.0
    path_mention: float = 0.0
    import_proximity: float = 0.0
    recency: float = 0.0

    def line_span(self) -> int:
        return max(0, self.end_line - self.start_line + 1)


class Chunk(BaseModel):
    """A ranked chunk included in a bundle."""

    path: str
    start_line: int
    end_line: int
    content: str
    score: float
    reason: str
    tokens: int


class ContextBundle(BaseModel):
    """Result of `get_context_bundle` / `expand_context`."""

    intent: Intent
    budget_used: int
    budget_limit: int
    chunks: list[Chunk]
    excluded_summary: str
    bundle_id: str
    expansions: int = 0


class TokenEstimate(BaseModel):
    """Result of `estimate_tokens`."""

    tokens: int
    method: str
