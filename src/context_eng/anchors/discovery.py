"""Runtime anchor path discovery (query + repo only, no oracle labels)."""

from __future__ import annotations

from pathlib import Path

from context_eng.config import Config
from context_eng.models import CandidateChunk, QueryAnalysis
from context_eng.retrieval.anchor_inference import infer_anchor_files
from context_eng.workspace import iter_files, relpath


def resolve_mentioned_files(
    mentions: list[str],
    workspace: Path,
    ignore_globs: tuple[str, ...],
) -> list[str]:
    """Map query file mentions to workspace-relative paths."""
    if not mentions:
        return []
    norm = [m.replace("\\", "/").lstrip("./") for m in mentions]
    resolved: dict[str, str] = {}
    for path in iter_files(workspace, ignore_globs):
        rel = relpath(path, workspace)
        name = path.name
        for mention in norm:
            if rel.endswith(mention) or name == mention.split("/")[-1]:
                resolved[rel] = rel
    return sorted(resolved.values())


def discover_anchor_paths(
    query: str,
    analysis: QueryAnalysis,
    workspace: Path,
    grep: list[CandidateChunk],
    config: Config,
) -> list[str]:
    """Paths treated as must-include anchors: explicit mentions + inferred files."""
    explicit = resolve_mentioned_files(
        analysis.signals.mentioned_files,
        workspace,
        config.ignore_globs,
    )
    explicit_set = set(explicit)
    inferred: list[str] = []
    if config.enable_anchor_inference and grep:
        inferred = [
            item.path
            for item in infer_anchor_files(
                query,
                grep,
                limit=config.max_inferred_anchor_files,
                min_score=config.inferred_anchor_min_score,
            )
            if item.path not in explicit_set
        ]
    return sorted(set(explicit) | set(inferred))
