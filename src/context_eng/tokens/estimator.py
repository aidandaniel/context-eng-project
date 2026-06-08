"""Token estimation.

Uses tiktoken (cl100k_base) when available for accuracy; otherwise falls back
to a chars/4 heuristic. The same estimator is used for both baseline and MCP
measurements so comparisons stay apples-to-apples.
"""

from __future__ import annotations

from functools import lru_cache

from context_eng.models import TokenEstimate

_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _encoder():
    """Return a cached tiktoken encoder, or None if tiktoken is unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Return an integer token count for ``text``."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN)


def method_name() -> str:
    return "tiktoken/cl100k_base" if _encoder() is not None else "chars/4"


def estimate(text: str) -> TokenEstimate:
    """Return a TokenEstimate for ``text``."""
    return TokenEstimate(tokens=count_tokens(text), method=method_name())
