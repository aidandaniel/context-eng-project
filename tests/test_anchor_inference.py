"""Tests for grounded anchor inference."""

from context_eng.models import CandidateChunk
from context_eng.retrieval.anchor_inference import infer_anchor_files


def _candidate(path: str, keyword_match: float) -> CandidateChunk:
    return CandidateChunk(
        path=path,
        start_line=1,
        end_line=3,
        content="token refresh",
        keyword_match=keyword_match,
    )


def test_infer_anchor_files_prefers_keyword_and_path_overlap():
    anchors = infer_anchor_files(
        "Why does token refresh fail?",
        [
            _candidate("src/auth/refresh.py", 2.0),
            _candidate("src/users/service.py", 1.0),
        ],
        limit=1,
        min_score=1.0,
    )

    assert [a.path for a in anchors] == ["src/auth/refresh.py"]
