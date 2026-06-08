"""Pure-Python keyword retriever with snippet extraction.

Uses ripgrep when available for speed, otherwise a Python scan (the MVP target
machine has no ``rg``, so the fallback is the primary path and is fully tested).
Candidate ``keyword_match`` is a raw hit count; normalization happens in the
ranker so all feature weighting lives in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

from context_eng.config import Config
from context_eng.models import CandidateChunk
from context_eng.workspace import iter_files, read_text, relpath

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "at", "is", "are", "was", "were", "be", "been", "this",
    "that", "it", "with", "from", "by", "as", "we", "i", "you", "my", "our",
    "fix", "add", "make", "use", "using", "how", "does", "do", "what", "why",
    "when", "where", "should", "would", "could", "please", "can", "after",
    "before", "into", "out", "up", "down", "not", "no", "yes",
}

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def extract_keywords(query: str, extra: list[str] | None = None) -> list[str]:
    """Derive search keywords from a query plus optional extra terms."""
    raw = _WORD_RE.findall(query)
    for term in extra or []:
        raw.extend(_WORD_RE.findall(term))

    keywords: list[str] = []
    for word in raw:
        lw = word.lower()
        if lw in _STOPWORDS:
            continue
        keywords.append(word)
        # Also include camelCase sub-tokens so "refreshToken" matches "refresh".
        parts = _CAMEL_SPLIT_RE.split(word)
        if len(parts) > 1:
            keywords.extend(p for p in parts if len(p) >= 3)

    # Dedupe case-insensitively, preserve order.
    seen: dict[str, str] = {}
    for kw in keywords:
        seen.setdefault(kw.lower(), kw)
    return list(seen.values())


def _merge_line_groups(
    hit_lines: list[int], context: int, total_lines: int
) -> list[tuple[int, int]]:
    """Merge nearby hit lines into (start, end) 1-based inclusive ranges."""
    if not hit_lines:
        return []
    hit_lines = sorted(set(hit_lines))
    groups: list[tuple[int, int]] = []
    start = max(1, hit_lines[0] - context)
    end = min(total_lines, hit_lines[0] + context)
    for ln in hit_lines[1:]:
        new_start = max(1, ln - context)
        if new_start <= end + 1:
            end = min(total_lines, ln + context)
        else:
            groups.append((start, end))
            start = new_start
            end = min(total_lines, ln + context)
    groups.append((start, end))
    return groups


class GrepRetriever:
    """Keyword retriever implementing the Retriever protocol."""

    def __init__(self, config: Config):
        self.config = config

    def search(
        self, query: str, workspace: Path, limit: int
    ) -> list[CandidateChunk]:
        keywords = extract_keywords(query)
        if not keywords:
            return []
        patterns = [(kw, re.compile(re.escape(kw), re.IGNORECASE)) for kw in keywords]
        context = self.config.grep_context_lines

        candidates: list[CandidateChunk] = []
        for path in iter_files(workspace, self.config.ignore_globs):
            source = read_text(path)
            if not source:
                continue
            lines = source.splitlines()
            hit_lines: list[int] = []
            for idx, line in enumerate(lines):
                if any(rx.search(line) for _, rx in patterns):
                    hit_lines.append(idx + 1)
            if not hit_lines:
                continue

            rel = relpath(path, workspace)
            for start, end in _merge_line_groups(hit_lines, context, len(lines)):
                snippet = "\n".join(lines[start - 1 : end])
                distinct = sum(1 for _, rx in patterns if rx.search(snippet))
                candidates.append(
                    CandidateChunk(
                        path=rel,
                        start_line=start,
                        end_line=end,
                        content=snippet,
                        tier="grep",
                        keyword_match=float(distinct),
                    )
                )

        candidates.sort(key=lambda c: c.keyword_match, reverse=True)
        return candidates[:limit]
