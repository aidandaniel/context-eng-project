"""Greedy, accuracy-preserving token budget packing.

Order of operations:
1. Must-include candidates (explicitly mentioned paths/symbols, stack-trace
   anchors) are added first, highest score first, even if they consume most of
   the budget. They are only bounded by a hard ceiling to avoid runaway.
2. Remaining budget is filled greedily by score.
Overlapping/duplicate ranges from the same file are de-duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass

from context_eng.models import Chunk
from context_eng.ranking.ranker import ScoredCandidate
from context_eng.tokens import count_tokens


@dataclass
class PackResult:
    chunks: list[Chunk]
    budget_used: int
    excluded_count: int


def _overlaps(a: Chunk, path: str, start: int, end: int) -> bool:
    return a.path == path and not (end < a.start_line or start > a.end_line)


def _is_dup(chosen: list[Chunk], path: str, start: int, end: int) -> bool:
    return any(_overlaps(c, path, start, end) for c in chosen)


def _to_chunk(sc: ScoredCandidate) -> Chunk:
    c = sc.candidate
    return Chunk(
        path=c.path,
        start_line=c.start_line,
        end_line=c.end_line,
        content=c.content,
        score=sc.score,
        reason=sc.reason,
        tokens=count_tokens(c.content),
    )


class BudgetPolicy:
    def __init__(self, hard_ceiling_factor: float = 1.5):
        # Must-include items may exceed the soft limit up to this factor of it.
        self.hard_ceiling_factor = hard_ceiling_factor

    def pack(
        self,
        scored: list[ScoredCandidate],
        budget_limit: int,
        must_include: set[int] | None = None,
    ) -> PackResult:
        must_include = must_include or set()
        ceiling = int(budget_limit * self.hard_ceiling_factor)

        must = [sc for i, sc in enumerate(scored) if i in must_include]
        rest = [sc for i, sc in enumerate(scored) if i not in must_include]
        must.sort(key=lambda s: s.score, reverse=True)

        chosen: list[Chunk] = []
        used = 0
        considered = 0

        for sc in must:
            considered += 1
            chunk = _to_chunk(sc)
            if _is_dup(chosen, chunk.path, chunk.start_line, chunk.end_line):
                continue
            if used + chunk.tokens > ceiling and chosen:
                continue
            chosen.append(chunk)
            used += chunk.tokens

        for sc in rest:
            considered += 1
            chunk = _to_chunk(sc)
            if _is_dup(chosen, chunk.path, chunk.start_line, chunk.end_line):
                continue
            if used + chunk.tokens > budget_limit:
                continue
            chosen.append(chunk)
            used += chunk.tokens

        excluded = len(scored) - len(chosen)
        return PackResult(chunks=chosen, budget_used=used, excluded_count=excluded)
