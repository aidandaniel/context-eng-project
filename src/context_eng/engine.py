"""Context engine: orchestrates analysis, retrieval, ranking, and packing.

This module contains all the logic and is independent of the MCP transport so
it can be unit-tested and driven by the benchmark harness directly.
"""

from __future__ import annotations

from pathlib import Path

from context_eng.budget.policy import BudgetPolicy
from context_eng.config import Config, load_config
from context_eng.intent import classifier
from context_eng.intent.budgets import budget_for
from context_eng.logging.store import EventLogger
from context_eng.models import (
    CandidateChunk,
    ContextBundle,
    Intent,
    QueryAnalysis,
    TokenEstimate,
)
from context_eng.ranking.ranker import ChunkRanker, RankWeights
from context_eng.retrieval.anchor_inference import infer_anchor_files
from context_eng.retrieval.grep_retriever import GrepRetriever, extract_keywords
from context_eng.retrieval.import_graph import local_imports
from context_eng.retrieval.symbol_slice import find_symbol_span
from context_eng.tokens import estimate
from context_eng.workspace import iter_files, read_text, relpath

_HEAD_LINES = 40
_IMPORT_HEAD_LINES = 18
_SKELETON_MAX_TOKENS = 500


class _BundleState:
    """Server-side memory for a bundle so expand_context can build on it."""

    __slots__ = (
        "query", "analysis", "budget_limit", "candidates", "expansions", "bundle"
    )

    def __init__(self, query, analysis, budget_limit, candidates):
        self.query = query
        self.analysis = analysis
        self.budget_limit = budget_limit
        self.candidates = candidates
        self.expansions = 0
        self.bundle: ContextBundle | None = None


class ContextEngine:
    def __init__(
        self,
        config: Config | None = None,
        weights: RankWeights | None = None,
    ):
        self.config = config or load_config()
        self.retriever = GrepRetriever(self.config)
        self.ranker = ChunkRanker(weights)
        self.policy = BudgetPolicy()
        self.logger = EventLogger(self.config.resolved_events_path)
        self._bundles: dict[str, _BundleState] = {}

    # ------------------------------------------------------------------ #
    # Public API (mirrors the MCP tools)
    # ------------------------------------------------------------------ #

    def analyze_query(self, query: str) -> QueryAnalysis:
        analysis = classifier.analyze(query, self.config)
        self.logger.log(
            {
                "event": "analyze_query",
                "query_id": EventLogger.new_id(),
                "query_tokens": analysis.signals.query_tokens,
                "intent": analysis.intent.value,
                "intent_confidence": analysis.confidence,
                "budget_assigned": analysis.budget.recommended,
                "features": {
                    "has_stack_trace": analysis.signals.has_stack_trace,
                    "mentioned_files": len(analysis.signals.mentioned_files),
                    "mentioned_symbols": len(analysis.signals.mentioned_symbols),
                },
            }
        )
        return analysis

    def estimate_tokens(self, text: str) -> TokenEstimate:
        return estimate(text)

    def estimate_bundle_tokens(self, bundle_id: str) -> TokenEstimate:
        state = self._bundles.get(bundle_id)
        if state is None or state.bundle is None:
            return TokenEstimate(tokens=0, method=estimate("").method)
        total = sum(c.tokens for c in state.bundle.chunks)
        return TokenEstimate(tokens=total, method=estimate("").method)

    def analysis_for_bundle(self, bundle_id: str) -> QueryAnalysis | None:
        """Return the analysis used to build a stored bundle."""
        state = self._bundles.get(bundle_id)
        return state.analysis if state is not None else None

    def get_context_bundle(
        self,
        query: str,
        max_tokens: int | None = None,
        intent: str | None = None,
    ) -> ContextBundle:
        workspace = self.config.workspace_root
        analysis = classifier.analyze(query, self.config)
        if intent:
            try:
                forced = Intent(intent)
                analysis.intent = forced
                analysis.budget = budget_for(forced, self.config)
            except ValueError:
                pass

        budget_limit = max_tokens or analysis.budget.recommended

        candidates = self._build_candidates(query, analysis, workspace)
        bundle = self._pack_and_build(
            query, analysis, budget_limit, candidates, expansions=0
        )

        state = _BundleState(query, analysis, budget_limit, candidates)
        state.bundle = bundle
        self._bundles[bundle.bundle_id] = state

        self._log_bundle(bundle, analysis, expansions=0, grep_hits=len(candidates))
        return bundle

    def expand_context(
        self,
        bundle_id: str,
        focus: str | None = None,
        extra_tokens: int | None = None,
    ) -> ContextBundle:
        state = self._bundles.get(bundle_id)
        if state is None:
            raise KeyError(f"unknown bundle_id: {bundle_id}")

        workspace = self.config.workspace_root
        state.expansions += 1
        extra = extra_tokens or int(state.budget_limit * 0.5)
        new_limit = state.budget_limit + extra
        state.budget_limit = new_limit

        # Relax: add a second import hop and (optionally) a full focused file.
        extra_candidates = self._expand_candidates(state, focus, workspace)
        # Merge, de-duping by (path, start, end).
        existing = {(c.path, c.start_line, c.end_line) for c in state.candidates}
        for c in extra_candidates:
            key = (c.path, c.start_line, c.end_line)
            if key not in existing:
                state.candidates.append(c)
                existing.add(key)

        bundle = self._pack_and_build(
            state.query,
            state.analysis,
            new_limit,
            state.candidates,
            expansions=state.expansions,
            bundle_id=bundle_id,
        )
        state.bundle = bundle
        self._log_bundle(
            bundle, state.analysis, expansions=state.expansions,
            grep_hits=len(state.candidates),
        )
        return bundle

    # ------------------------------------------------------------------ #
    # Candidate generation
    # ------------------------------------------------------------------ #

    def _build_candidates(
        self,
        query: str,
        analysis: QueryAnalysis,
        workspace: Path,
    ) -> list[CandidateChunk]:
        candidates: list[CandidateChunk] = []

        anchor_files = self._resolve_mentioned_files(
            analysis.signals.mentioned_files, workspace
        )
        grep = self.retriever.search(
            query, workspace, self.config.max_grep_candidates
        )

        inferred_anchor_files: list[Path] = []
        if not anchor_files and self.config.enable_anchor_inference:
            inferred = infer_anchor_files(
                query,
                grep,
                limit=self.config.max_inferred_anchor_files,
                min_score=self.config.inferred_anchor_min_score,
            )
            analysis.signals.inferred_files = [item.path for item in inferred]
            inferred_anchor_files = [workspace / item.path for item in inferred]

        # Tier 1+2: anchors and symbol slices.
        for path in anchor_files:
            candidates.extend(
                self._anchor_candidates(
                    path, analysis.signals.mentioned_symbols, workspace
                )
            )
        for path in inferred_anchor_files:
            candidates.extend(self._inferred_anchor_candidates(path, workspace))

        # Tier 2 (global): symbol definitions anywhere in the workspace.
        if analysis.signals.mentioned_symbols:
            candidates.extend(
                self._symbol_candidates(
                    analysis.signals.mentioned_symbols, workspace, anchor_files
                )
            )

        # Tier 3: 1-hop import neighbors of anchors.
        candidates.extend(
            self._import_candidates(anchor_files + inferred_anchor_files, workspace)
        )

        # Tier 4: grep hits.
        candidates.extend(grep)

        # Feature post-processing: path_mention + recency.
        self._apply_features(candidates, anchor_files + inferred_anchor_files, workspace)
        return candidates

    def _resolve_mentioned_files(
        self, mentions: list[str], workspace: Path
    ) -> list[Path]:
        if not mentions:
            return []
        norm = [m.replace("\\", "/").lstrip("./") for m in mentions]
        resolved: dict[str, Path] = {}
        for path in iter_files(workspace, self.config.ignore_globs):
            rel = relpath(path, workspace)
            name = path.name
            for m in norm:
                if rel.endswith(m) or name == m.split("/")[-1]:
                    resolved[rel] = path
        return list(resolved.values())

    def _anchor_candidates(
        self, path: Path, symbols: list[str], workspace: Path
    ) -> list[CandidateChunk]:
        source = read_text(path)
        if not source:
            return []
        rel = relpath(path, workspace)
        lines = source.splitlines()
        out: list[CandidateChunk] = []

        matched_symbol = False
        for sym in symbols:
            span = find_symbol_span(source, path.name, sym)
            if span is not None:
                snippet = "\n".join(lines[span.start_line - 1 : span.end_line])
                out.append(
                    CandidateChunk(
                        path=rel,
                        start_line=span.start_line,
                        end_line=span.end_line,
                        content=snippet,
                        tier="symbol",
                        keyword_match=1.0,
                        path_mention=1.0,
                    )
                )
                matched_symbol = True

        if not matched_symbol:
            end = min(len(lines), _HEAD_LINES)
            snippet = "\n".join(lines[:end])
            out.append(
                CandidateChunk(
                    path=rel,
                    start_line=1,
                    end_line=max(1, end),
                    content=snippet,
                    tier="anchor",
                    path_mention=1.0,
                )
            )
        return out

    def _inferred_anchor_candidates(
        self, path: Path, workspace: Path
    ) -> list[CandidateChunk]:
        source = read_text(path)
        if not source:
            return []
        rel = relpath(path, workspace)
        lines = source.splitlines()
        end = min(len(lines), _HEAD_LINES)
        return [
            CandidateChunk(
                path=rel,
                start_line=1,
                end_line=max(1, end),
                content="\n".join(lines[:end]),
                tier="inferred_anchor",
                path_mention=0.75,
            )
        ]

    def _symbol_candidates(
        self, symbols: list[str], workspace: Path, anchor_files: list[Path]
    ) -> list[CandidateChunk]:
        anchor_set = {p.resolve() for p in anchor_files}
        out: list[CandidateChunk] = []
        for path in iter_files(workspace, self.config.ignore_globs):
            if path.resolve() in anchor_set:
                continue
            source = read_text(path)
            if not source:
                continue
            rel = relpath(path, workspace)
            lines = source.splitlines()
            for sym in symbols:
                span = find_symbol_span(source, path.name, sym)
                if span is not None:
                    snippet = "\n".join(
                        lines[span.start_line - 1 : span.end_line]
                    )
                    out.append(
                        CandidateChunk(
                            path=rel,
                            start_line=span.start_line,
                            end_line=span.end_line,
                            content=snippet,
                            tier="symbol",
                            keyword_match=1.0,
                            import_proximity=0.5,
                        )
                    )
        return out

    def _import_candidates(
        self, anchor_files: list[Path], workspace: Path
    ) -> list[CandidateChunk]:
        out: list[CandidateChunk] = []
        seen: set[str] = set()
        for anchor in anchor_files:
            for neighbor in local_imports(anchor, workspace):
                rel = relpath(neighbor, workspace)
                if rel in seen:
                    continue
                seen.add(rel)
                source = read_text(neighbor)
                if not source:
                    continue
                lines = source.splitlines()
                end = min(len(lines), _IMPORT_HEAD_LINES)
                out.append(
                    CandidateChunk(
                        path=rel,
                        start_line=1,
                        end_line=max(1, end),
                        content="\n".join(lines[:end]),
                        tier="import",
                        import_proximity=0.7,
                    )
                )
        return out

    def _apply_features(
        self,
        candidates: list[CandidateChunk],
        anchor_files: list[Path],
        workspace: Path,
    ) -> None:
        anchor_rels = {relpath(p, workspace) for p in anchor_files}
        # recency: normalize mtime across candidate files.
        mtimes: dict[str, float] = {}
        for c in candidates:
            if c.path in mtimes:
                continue
            fpath = workspace / c.path
            try:
                mtimes[c.path] = fpath.stat().st_mtime
            except OSError:
                mtimes[c.path] = 0.0
        if mtimes:
            lo = min(mtimes.values())
            hi = max(mtimes.values())
            span = hi - lo or 1.0
        for c in candidates:
            # Boost (but do not force-include) other chunks from an anchor file.
            # The dedicated anchor/symbol slice already carries path_mention=1.0
            # and guarantees the file appears; extra snippets stay optional so
            # the whole anchor file is not pulled in piece by piece.
            if c.path in anchor_rels and c.path_mention == 0.0:
                c.path_mention = 0.5
            if mtimes:
                c.recency = (mtimes.get(c.path, lo) - lo) / span

    def _expand_candidates(
        self, state: _BundleState, focus: str | None, workspace: Path
    ) -> list[CandidateChunk]:
        out: list[CandidateChunk] = []
        # Full file for an explicit focus path.
        if focus:
            for path in self._resolve_mentioned_files([focus], workspace):
                source = read_text(path)
                if not source:
                    continue
                rel = relpath(path, workspace)
                lines = source.splitlines()
                out.append(
                    CandidateChunk(
                        path=rel,
                        start_line=1,
                        end_line=max(1, len(lines)),
                        content=source,
                        tier="anchor",
                        path_mention=1.0,
                    )
                )
        # Second import hop from files already in the candidate set.
        current_paths = {c.path for c in state.candidates}
        for rel in list(current_paths):
            fpath = workspace / rel
            if not fpath.is_file():
                continue
            for neighbor in local_imports(fpath, workspace):
                nrel = relpath(neighbor, workspace)
                if nrel in current_paths:
                    continue
                source = read_text(neighbor)
                if not source:
                    continue
                lines = source.splitlines()
                end = min(len(lines), _HEAD_LINES)
                out.append(
                    CandidateChunk(
                        path=nrel,
                        start_line=1,
                        end_line=max(1, end),
                        content="\n".join(lines[:end]),
                        tier="import",
                        import_proximity=0.4,
                    )
                )
        self._apply_features(out, [], workspace)
        return out

    # ------------------------------------------------------------------ #
    # Packing / assembly
    # ------------------------------------------------------------------ #

    def _pack_and_build(
        self,
        query: str,
        analysis: QueryAnalysis,
        budget_limit: int,
        candidates: list[CandidateChunk],
        expansions: int,
        bundle_id: str | None = None,
    ) -> ContextBundle:
        scored = self.ranker.rank(candidates)

        # Partition into must-include anchors vs optional chunks. Optional
        # chunks below the score threshold are dropped (budget is a ceiling,
        # not a fill target), and the rest are capped so a broad query cannot
        # pad the bundle with marginal snippets.
        kept: list = []
        optional_kept = 0
        must_indices: set[int] = set()
        for sc in scored:
            cand = sc.candidate
            is_must = cand.path_mention >= 1.0 or cand.tier in (
                "anchor",
                "symbol",
                "inferred_anchor",
            )
            if is_must:
                must_indices.add(len(kept))
                kept.append(sc)
                continue
            if sc.score < self.config.min_chunk_score:
                continue
            if optional_kept >= self.config.max_optional_chunks:
                continue
            optional_kept += 1
            kept.append(sc)

        dropped_by_filter = len(scored) - len(kept)
        result = self.policy.pack(kept, budget_limit, must_indices)
        result.excluded_count += dropped_by_filter

        excluded_summary = (
            f"{result.excluded_count} candidates dropped "
            f"(below score threshold or over budget)"
        )
        return ContextBundle(
            intent=analysis.intent,
            budget_used=result.budget_used,
            budget_limit=budget_limit,
            chunks=result.chunks,
            excluded_summary=excluded_summary,
            bundle_id=bundle_id or EventLogger.new_id(),
            expansions=expansions,
        )

    def _log_bundle(
        self,
        bundle: ContextBundle,
        analysis: QueryAnalysis,
        expansions: int,
        grep_hits: int,
    ) -> None:
        self.logger.log(
            {
                "event": "get_context_bundle",
                "query_id": bundle.bundle_id,
                "query_tokens": analysis.signals.query_tokens,
                "intent": bundle.intent.value,
                "intent_confidence": analysis.confidence,
                "budget_assigned": bundle.budget_limit,
                "budget_used": bundle.budget_used,
                "chunk_count": len(bundle.chunks),
                "expansions": expansions,
                "features": {
                    "has_stack_trace": analysis.signals.has_stack_trace,
                    "mentioned_files": len(analysis.signals.mentioned_files),
                    "inferred_files": len(analysis.signals.inferred_files),
                    "grep_hits": grep_hits,
                },
                "success": None,
            }
        )
