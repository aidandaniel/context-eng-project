"""Tests for context bundle formatting."""

from context_eng.formatting import format_context_message
from context_eng.models import (
    BudgetInfo,
    Chunk,
    ContextBundle,
    Intent,
    QueryAnalysis,
    QuerySignals,
)


def test_format_context_message_includes_chunks():
    analysis = QueryAnalysis(
        intent=Intent.EXPLAIN,
        confidence=0.8,
        signals=QuerySignals(mentioned_files=["auth.py"], query_tokens=12),
        budget=BudgetInfo(recommended=4000, min=2000, max=6000),
    )
    bundle = ContextBundle(
        intent=Intent.EXPLAIN,
        budget_used=120,
        budget_limit=4000,
        chunks=[
            Chunk(
                path="auth.py",
                start_line=1,
                end_line=5,
                content="def login():\n    pass",
                score=0.9,
                reason="anchor file",
                tokens=20,
            )
        ],
        excluded_summary="",
        bundle_id="test-bundle",
    )

    text = format_context_message("how does login work?", analysis, bundle, "/repo")
    assert "how does login work?" in text
    assert "auth.py" in text
    assert "test-bundle" in text
    assert "def login():" in text
