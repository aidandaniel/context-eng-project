"""Weighted chunk ranking.

score = 0.4*keyword_match + 0.3*path_mention + 0.2*import_proximity + 0.1*recency

``keyword_match`` arrives as a raw hit count and is min-max normalized across
the candidate set here. The other features are expected to already be in 0..1
(set by the engine). Weights are configurable so the benchmark can tune them.
"""

from __future__ import annotations

from dataclasses import dataclass

from context_eng.models import CandidateChunk


@dataclass(frozen=True)
class RankWeights:
    keyword_match: float = 0.4
    path_mention: float = 0.3
    import_proximity: float = 0.2
    recency: float = 0.1


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: CandidateChunk
    score: float
    reason: str


def _normalize_keyword(values: list[float]) -> list[float]:
    if not values:
        return []
    hi = max(values)
    if hi <= 0:
        return [0.0 for _ in values]
    return [v / hi for v in values]


def _reason(c: CandidateChunk) -> str:
    parts: list[str] = []
    if c.path_mention > 0:
        parts.append("explicitly mentioned path")
    if c.tier == "symbol":
        parts.append("symbol slice")
    elif c.tier == "import":
        parts.append("import neighbor")
    elif c.tier == "skeleton":
        parts.append("project skeleton")
    if c.keyword_match > 0 and c.tier == "grep":
        parts.append("keyword match")
    if c.import_proximity > 0 and c.tier != "import":
        parts.append("near anchor in import graph")
    return ", ".join(parts) or "candidate"


class ChunkRanker:
    def __init__(self, weights: RankWeights | None = None):
        self.weights = weights or RankWeights()

    def rank(self, candidates: list[CandidateChunk]) -> list[ScoredCandidate]:
        if not candidates:
            return []
        norm_kw = _normalize_keyword([c.keyword_match for c in candidates])
        w = self.weights
        scored: list[ScoredCandidate] = []
        for c, kw in zip(candidates, norm_kw):
            score = (
                w.keyword_match * kw
                + w.path_mention * min(1.0, c.path_mention)
                + w.import_proximity * min(1.0, c.import_proximity)
                + w.recency * min(1.0, c.recency)
            )
            scored.append(ScoredCandidate(c, round(score, 4), _reason(c)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored
