"""Unit tests for the rule-based intent classifier and signal extraction."""

import pytest

from context_eng.intent import classifier
from context_eng.intent.budgets import budget_for
from context_eng.models import Intent


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Fix the TypeError thrown in auth/refresh.py", Intent.DEBUG),
        ("Traceback (most recent call last): NameError", Intent.DEBUG),
        ("Add a new endpoint to create users", Intent.IMPLEMENT),
        ("How does the ranker compute its score?", Intent.EXPLAIN),
        ("Refactor and rename the helper function", Intent.REFACTOR),
        ("Please review this PR for security issues", Intent.REVIEW),
    ],
)
def test_intent_classification(query, expected):
    analysis = classifier.analyze(query)
    assert analysis.intent == expected
    assert 0.0 <= analysis.confidence <= 1.0


def test_unknown_query_defaults_to_implement_low_confidence():
    analysis = classifier.analyze("the quick brown fox jumps over stuff")
    assert analysis.intent == Intent.IMPLEMENT
    assert analysis.confidence <= 0.5


def test_signals_extract_files_and_symbols():
    signals = classifier.extract_signals(
        "Fix `refreshToken` in src/auth/refresh.py and middleware.ts"
    )
    assert "src/auth/refresh.py" in signals.mentioned_files
    assert "middleware.ts" in signals.mentioned_files
    assert "refreshToken" in signals.mentioned_symbols
    assert signals.query_tokens > 0


def test_stack_trace_detection_boosts_debug():
    signals = classifier.extract_signals("ValueError: bad input")
    assert signals.has_stack_trace is True
    assert signals.has_error_token is True


def test_budget_table_has_clamps():
    info = budget_for(Intent.DEBUG)
    assert info.min <= info.recommended <= info.max
    assert info.source == "fixed_intent_table"
