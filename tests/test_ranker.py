"""Unit tests for the chunk ranker."""

from context_eng.models import CandidateChunk
from context_eng.ranking.ranker import ChunkRanker, RankWeights


def _chunk(path, kw=0.0, pm=0.0, ip=0.0, rec=0.0, tier="grep"):
    return CandidateChunk(
        path=path,
        start_line=1,
        end_line=10,
        content="x" * 20,
        tier=tier,
        keyword_match=kw,
        path_mention=pm,
        import_proximity=ip,
        recency=rec,
    )


def test_ranker_orders_by_weighted_score():
    candidates = [
        _chunk("low.py", kw=1.0),
        _chunk("high.py", kw=5.0, pm=1.0),
        _chunk("mid.py", kw=3.0),
    ]
    ranked = ChunkRanker().rank(candidates)
    assert [r.candidate.path for r in ranked][0] == "high.py"
    # Scores are sorted descending.
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_path_mention_outweighs_weak_keyword():
    candidates = [
        _chunk("mentioned.py", kw=1.0, pm=1.0),
        _chunk("other.py", kw=2.0, pm=0.0),
    ]
    ranked = ChunkRanker().rank(candidates)
    # mentioned.py: 0.4*0.5 + 0.3*1 = 0.5 ; other.py: 0.4*1 = 0.4
    assert ranked[0].candidate.path == "mentioned.py"


def test_keyword_normalization_is_relative():
    candidates = [_chunk("a.py", kw=10.0), _chunk("b.py", kw=5.0)]
    ranked = ChunkRanker(RankWeights(keyword_match=1.0, path_mention=0,
                                     import_proximity=0, recency=0)).rank(candidates)
    assert ranked[0].score == 1.0  # max normalized to 1
    assert ranked[1].score == 0.5


def test_empty_candidates():
    assert ChunkRanker().rank([]) == []


def test_reason_mentions_tier():
    ranked = ChunkRanker().rank([_chunk("s.py", tier="symbol", kw=1.0)])
    assert "symbol slice" in ranked[0].reason
