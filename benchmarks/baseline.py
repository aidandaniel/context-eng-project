"""Baseline (before-MCP) context gathering strategies.

These simulate how an agent without context engineering pulls context: grep the
query keywords and read whole files. Token counts use the same estimator as the
MCP path so the comparison is fair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from context_eng.config import Config
from context_eng.retrieval.grep_retriever import extract_keywords
from context_eng.retrieval.import_graph import local_imports
from context_eng.tokens import count_tokens
from context_eng.workspace import iter_files, read_text, relpath


@dataclass
class BaselineResult:
    strategy: str
    files: list[str]
    tokens: int


def _keyword_hit_counts(
    query: str, config: Config
) -> list[tuple[str, int, str]]:
    """Return (relpath, distinct_keyword_hits, content) sorted by hits desc."""
    keywords = extract_keywords(query)
    patterns = [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]
    scored: list[tuple[str, int, str]] = []
    for path in iter_files(config.workspace_root, config.ignore_globs):
        content = read_text(path)
        if not content:
            continue
        distinct = sum(1 for rx in patterns if rx.search(content))
        if distinct == 0:
            continue
        scored.append((relpath(path, config.workspace_root), distinct, content))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


def grep_top_k_full_files(query: str, config: Config, k: int) -> BaselineResult:
    scored = _keyword_hit_counts(query, config)
    top = scored[:k]
    files = [rel for rel, _, _ in top]
    tokens = sum(count_tokens(content) for _, _, content in top)
    return BaselineResult("grep_top_k_full_files", files, tokens)


def all_grep_hits(query: str, config: Config, k: int = 0) -> BaselineResult:
    scored = _keyword_hit_counts(query, config)
    files = [rel for rel, _, _ in scored]
    tokens = sum(count_tokens(content) for _, _, content in scored)
    return BaselineResult("all_grep_hits", files, tokens)


def mentioned_plus_imports(query: str, config: Config, k: int = 0) -> BaselineResult:
    """Read explicitly mentioned files plus their 1-hop imports, in full."""
    from context_eng.intent.classifier import extract_signals

    workspace = config.workspace_root
    signals = extract_signals(query)
    mentions = [m.replace("\\", "/").lstrip("./") for m in signals.mentioned_files]

    chosen: dict[str, Path] = {}
    for path in iter_files(workspace, config.ignore_globs):
        rel = relpath(path, workspace)
        if any(rel.endswith(m) for m in mentions):
            chosen[rel] = path
            for neighbor in local_imports(path, workspace):
                chosen[relpath(neighbor, workspace)] = neighbor

    tokens = sum(count_tokens(read_text(p)) for p in chosen.values())
    return BaselineResult("mentioned_plus_imports", list(chosen), tokens)


STRATEGIES = {
    "grep_top_k_full_files": grep_top_k_full_files,
    "all_grep_hits": all_grep_hits,
    "mentioned_plus_imports": mentioned_plus_imports,
}


def run_baseline(
    query: str, config: Config, strategy: str, k: int
) -> BaselineResult:
    fn = STRATEGIES.get(strategy, grep_top_k_full_files)
    return fn(query, config, k)
