"""Unit + integration tests for budget packing and bundle assembly."""

import textwrap

import pytest

from context_eng.budget.policy import BudgetPolicy
from context_eng.config import Config
from context_eng.engine import ContextEngine
from context_eng.models import CandidateChunk
from context_eng.ranking.ranker import ChunkRanker
from context_eng.retrieval.symbol_slice import find_symbol_span


def _scored(path, tokens_content, score_kw=1.0, pm=0.0, tier="grep"):
    return CandidateChunk(
        path=path,
        start_line=1,
        end_line=5,
        content=tokens_content,
        tier=tier,
        keyword_match=score_kw,
        path_mention=pm,
    )


def test_budget_policy_respects_limit():
    cands = [_scored(f"f{i}.py", "word " * 100, score_kw=float(10 - i)) for i in range(10)]
    scored = ChunkRanker().rank(cands)
    result = BudgetPolicy().pack(scored, budget_limit=120)
    assert result.budget_used <= 120
    assert result.excluded_count > 0


def test_budget_policy_must_include_anchor_even_if_low_score():
    anchor = _scored("anchor.py", "anchor " * 50, score_kw=0.0, pm=1.0, tier="anchor")
    filler = [_scored(f"f{i}.py", "word " * 50, score_kw=float(5 - i)) for i in range(5)]
    cands = filler + [anchor]
    scored = ChunkRanker().rank(cands)
    must = {i for i, sc in enumerate(scored) if sc.candidate.path == "anchor.py"}
    result = BudgetPolicy().pack(scored, budget_limit=10, must_include=must)
    assert any(c.path == "anchor.py" for c in result.chunks)


def test_budget_policy_dedups_overlapping_ranges():
    a = CandidateChunk(path="x.py", start_line=1, end_line=20, content="a " * 10,
                       tier="grep", keyword_match=2.0)
    b = CandidateChunk(path="x.py", start_line=5, end_line=15, content="b " * 10,
                       tier="grep", keyword_match=1.0)
    scored = ChunkRanker().rank([a, b])
    result = BudgetPolicy().pack(scored, budget_limit=10_000)
    assert len(result.chunks) == 1  # overlapping range dropped


def test_symbol_slice_python():
    source = textwrap.dedent(
        '''
        def unrelated():
            return 1

        def refresh_token(user):
            token = make_token(user)
            return token
        '''
    ).strip()
    span = find_symbol_span(source, "refresh.py", "refresh_token")
    assert span is not None
    assert span.start_line < span.end_line


@pytest.fixture
def sample_repo(tmp_path):
    auth = tmp_path / "src" / "auth"
    auth.mkdir(parents=True)
    (auth / "refresh.py").write_text(
        textwrap.dedent(
            '''
            from src.auth.tokens import make_token

            def refreshToken(user):
                """Refresh a session token."""
                return make_token(user)

            def logout(user):
                user.session = None
            ''' + "\n# padding\n" * 200
        ),
        encoding="utf-8",
    )
    (auth / "tokens.py").write_text(
        "def make_token(user):\n    return 'tok'\n", encoding="utf-8"
    )
    noise = tmp_path / "src" / "noise"
    noise.mkdir(parents=True)
    for i in range(5):
        (noise / f"mod{i}.py").write_text(
            "def thing():\n    return %d\n" % i + "# filler\n" * 300,
            encoding="utf-8",
        )
    return tmp_path


def test_bundle_includes_anchor_and_stays_in_budget(sample_repo):
    cfg = Config(workspace_root=sample_repo)
    engine = ContextEngine(config=cfg)
    bundle = engine.get_context_bundle(
        "Fix TypeError in refreshToken in src/auth/refresh.py"
    )
    paths = {c.path for c in bundle.chunks}
    assert "src/auth/refresh.py" in paths
    # Anchor must-include is bounded by the policy ceiling (1.5x).
    assert bundle.budget_used <= int(bundle.budget_limit * 1.5)


def test_bundle_symbol_slice_smaller_than_full_file(sample_repo):
    cfg = Config(workspace_root=sample_repo)
    engine = ContextEngine(config=cfg)
    bundle = engine.get_context_bundle(
        "Fix TypeError in refreshToken in src/auth/refresh.py"
    )
    full = (sample_repo / "src" / "auth" / "refresh.py").read_text(encoding="utf-8")
    refresh_chunks = [c for c in bundle.chunks if c.path == "src/auth/refresh.py"]
    assert refresh_chunks
    # The slice should be far smaller than the whole padded file.
    assert max(c.end_line - c.start_line + 1 for c in refresh_chunks) < len(
        full.splitlines()
    )


def test_expand_context_grows_bundle(sample_repo):
    cfg = Config(workspace_root=sample_repo)
    engine = ContextEngine(config=cfg)
    bundle = engine.get_context_bundle("refreshToken logout session")
    expanded = engine.expand_context(bundle.bundle_id)
    assert expanded.expansions == 1
    assert expanded.budget_limit > bundle.budget_limit


def test_bundle_infers_anchor_when_query_has_no_file(sample_repo):
    cfg = Config(workspace_root=sample_repo)
    engine = ContextEngine(config=cfg)
    bundle = engine.get_context_bundle("Why does refreshToken fail after logout?")

    assert "src/auth/refresh.py" in {c.path for c in bundle.chunks}
