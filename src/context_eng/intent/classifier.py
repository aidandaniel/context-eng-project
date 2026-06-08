"""Rule-based query intent classifier and signal extraction.

Deliberately dependency-free and deterministic so it is fast, testable, and
produces stable features for the event log (future ML training data).
"""

from __future__ import annotations

import re

from context_eng.config import Config
from context_eng.intent.budgets import budget_for
from context_eng.models import Intent, QueryAnalysis, QuerySignals
from context_eng.tokens import count_tokens

# Keyword buckets per intent. Order matters only for tie-breaking (see _INTENT_ORDER).
_INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.DEBUG: (
        "fix", "bug", "error", "exception", "traceback", "fail", "failing",
        "crash", "broken", "stack trace", "throws", "regression", "not working",
    ),
    Intent.IMPLEMENT: (
        "add", "create", "implement", "build", "introduce", "new feature",
        "support", "write a", "generate", "scaffold",
    ),
    Intent.EXPLAIN: (
        "how does", "what is", "explain", "why does", "walk me through",
        "understand", "describe", "what happens",
    ),
    Intent.REFACTOR: (
        "refactor", "rename", "extract", "move", "clean up", "simplify",
        "restructure", "deduplicate", "split", "inline",
    ),
    Intent.REVIEW: (
        "review", "pr", "pull request", "audit", "security", "vulnerab",
        "lgtm", "code smell", "best practice",
    ),
}

# Tie-break priority: debug wins over the rest when scores tie.
_INTENT_ORDER: tuple[Intent, ...] = (
    Intent.DEBUG,
    Intent.REFACTOR,
    Intent.REVIEW,
    Intent.EXPLAIN,
    Intent.IMPLEMENT,
)

_CODE_EXTENSIONS = (
    "py", "pyi", "ts", "tsx", "js", "jsx", "mjs", "cjs", "java", "go", "rs",
    "rb", "php", "c", "h", "cpp", "hpp", "cs", "kt", "swift", "scala", "sql",
    "json", "toml", "yaml", "yml", "md",
)

_FILE_RE = re.compile(
    r"[\w./\\-]+\.(?:" + "|".join(_CODE_EXTENSIONS) + r")\b",
    re.IGNORECASE,
)

# Backticked tokens, camelCase, or snake_case identifiers (len >= 3).
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_CAMEL_RE = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]+\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

_STACK_TRACE_RE = re.compile(
    r"(traceback \(most recent call last\)|"
    r'file "[^"]+", line \d+|'
    r"\bat [\w./\\-]+:\d+|"
    r"\b\w*error\b|\bexception\b)",
    re.IGNORECASE,
)
_ERROR_TOKEN_RE = re.compile(
    r"\b(error|exception|traceback|typeerror|valueerror|keyerror|"
    r"nullpointer|undefined|nameerror|attributeerror)\b",
    re.IGNORECASE,
)


def extract_signals(query: str) -> QuerySignals:
    lowered = query.lower()

    files = [m.group(0).replace("\\", "/") for m in _FILE_RE.finditer(query)]
    files = list(dict.fromkeys(files))

    symbols: list[str] = []
    for m in _BACKTICK_RE.finditer(query):
        token = m.group(1).strip()
        # Skip backticked file paths (already captured) and multi-word phrases.
        if token and " " not in token and not _FILE_RE.fullmatch(token):
            symbols.append(token)
    symbols.extend(_CAMEL_RE.findall(query))
    symbols.extend(s for s in _SNAKE_RE.findall(query) if len(s) >= 3)
    # Drop anything that is actually a captured file path.
    file_set = set(files)
    symbols = [s for s in dict.fromkeys(symbols) if s not in file_set]

    return QuerySignals(
        has_stack_trace=bool(_STACK_TRACE_RE.search(lowered)),
        mentioned_files=files,
        mentioned_symbols=symbols,
        has_error_token=bool(_ERROR_TOKEN_RE.search(lowered)),
        query_tokens=count_tokens(query),
    )


def _score_intents(query: str) -> dict[Intent, int]:
    lowered = query.lower()
    scores: dict[Intent, int] = {intent: 0 for intent in Intent}
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lowered:
                scores[intent] += 1
    return scores


def classify(query: str, signals: QuerySignals) -> tuple[Intent, float]:
    """Return (intent, confidence) for ``query``."""
    scores = _score_intents(query)

    # Strong structural signal: a stack trace strongly implies debugging.
    if signals.has_stack_trace:
        scores[Intent.DEBUG] += 2

    best = max(_INTENT_ORDER, key=lambda i: (scores[i], -_INTENT_ORDER.index(i)))
    best_score = scores[best]

    if best_score == 0:
        # Nothing matched: default to implement at low confidence.
        return Intent.IMPLEMENT, 0.3

    total = sum(scores.values())
    # Confidence: dominance of the winning bucket, floored so a clear single
    # match still reads as reasonably confident.
    confidence = best_score / total if total else 0.0
    confidence = max(0.4, min(0.99, confidence))
    return best, round(confidence, 2)


def analyze(query: str, config: Config | None = None) -> QueryAnalysis:
    """Full analysis: signals + intent + budget."""
    signals = extract_signals(query)
    intent, confidence = classify(query, signals)
    budget = budget_for(intent, config)
    return QueryAnalysis(
        intent=intent,
        confidence=confidence,
        signals=signals,
        budget=budget,
    )
