"""Infer likely anchor files from grounded retrieval candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from context_eng.models import CandidateChunk
from context_eng.retrieval.grep_retriever import extract_keywords

_PATH_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


@dataclass(frozen=True)
class InferredAnchor:
    """A likely anchor path selected from files that actually exist."""

    path: str
    score: float
    reason: str


@dataclass(frozen=True)
class AnchorOption:
    """A grounded candidate file that an LLM may choose as an anchor."""

    path: str
    score: float
    reason: str
    evidence: str


def _path_tokens(path: str) -> set[str]:
    stem_parts = PurePosixPath(path).with_suffix("").parts
    tokens: set[str] = set()
    for part in stem_parts:
        tokens.update(t.lower() for t in _PATH_TOKEN_RE.findall(part))
    return tokens


def rank_anchor_options(
    query: str,
    candidates: list[CandidateChunk],
    *,
    limit: int,
    min_score: float,
) -> list[AnchorOption]:
    """Rank candidate files that could become inferred anchors.

    This is intentionally grounded: it can only choose paths that retrieval
    already found in the workspace, which keeps the FastMCP UX config-free and
    prevents invented paths from entering the bundle.
    """
    if not candidates or limit <= 0:
        return []

    query_terms = {kw.lower() for kw in extract_keywords(query)}
    grouped: dict[str, dict[str, object]] = {}
    for cand in candidates:
        stats = grouped.setdefault(
            cand.path, {"keyword": 0.0, "spans": 0.0, "evidence": ""}
        )
        if cand.keyword_match > float(stats["keyword"]):
            stats["keyword"] = cand.keyword_match
            stats["evidence"] = _compact_evidence(cand.content)
        stats["spans"] = float(stats["spans"]) + 1.0

    options: list[AnchorOption] = []
    for path, stats in grouped.items():
        path_overlap = len(query_terms & _path_tokens(path))
        score = (
            float(stats["keyword"])
            + (path_overlap * 1.5)
            + min(float(stats["spans"]), 3.0) * 0.25
        )
        if score < min_score:
            continue
        reasons: list[str] = []
        if stats["keyword"]:
            reasons.append("query keyword matches")
        if path_overlap:
            reasons.append("path terms match query")
        options.append(
            AnchorOption(
                path=path,
                score=round(score, 4),
                reason=", ".join(reasons) or "retrieval candidate",
                evidence=str(stats["evidence"]),
            )
        )

    options.sort(key=lambda item: (-item.score, item.path))
    return options[:limit]


def infer_anchor_files(
    query: str,
    candidates: list[CandidateChunk],
    *,
    limit: int,
    min_score: float,
) -> list[InferredAnchor]:
    """Select likely anchor files from grep candidates."""
    return [
        InferredAnchor(path=item.path, score=item.score, reason=item.reason)
        for item in rank_anchor_options(
            query, candidates, limit=limit, min_score=min_score
        )
    ]


def _compact_evidence(content: str, limit: int = 220) -> str:
    text = " ".join(line.strip() for line in content.splitlines() if line.strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
