"""FastMCP server exposing the Context Engineering tools.

The MCP layer is intentionally thin: it validates inputs and delegates to
``ContextEngine``. Run with ``python -m context_eng.server`` (stdio transport).

Multi-project support: pass ``workspace_root`` on each call (recommended when
using a global MCP config), or rely on ``CONTEXT_ENG_WORKSPACE`` / process cwd.
Engines are cached per workspace; bundles are tracked globally by ``bundle_id``
so ``expand_context`` works without re-passing the workspace.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from context_eng.config import load_config
from context_eng.engine import ContextEngine
from context_eng.formatting import format_context_message
from context_eng.workspace_resolve import resolve_workspace

mcp = FastMCP("context-eng")

# One engine per resolved workspace path (posix key).
_engines: dict[str, ContextEngine] = {}
# bundle_id -> engine that created it (for expand_context / estimate_tokens).
_bundle_owners: dict[str, ContextEngine] = {}


def get_engine(workspace_root: str | None = None) -> ContextEngine:
    """Return a cached ContextEngine for ``workspace_root`` (or cwd/env default)."""
    root = resolve_workspace(workspace_root)
    key = root.as_posix()
    if key not in _engines:
        _engines[key] = ContextEngine(config=load_config(str(root)))
    return _engines[key]


def _engine_for_bundle(bundle_id: str) -> ContextEngine | None:
    return _bundle_owners.get(bundle_id)


def _track_bundle(engine: ContextEngine, bundle_id: str) -> None:
    _bundle_owners[bundle_id] = engine


def _prepare_context(
    query: str,
    max_tokens: Optional[int] = None,
    intent: Optional[str] = None,
    workspace_root: Optional[str] = None,
) -> dict:
    """Analyze a query and return a ready-to-use context bundle."""
    engine = get_engine(workspace_root)
    bundle = engine.get_context_bundle(query, max_tokens, intent)
    analysis = engine.analysis_for_bundle(bundle.bundle_id) or engine.analyze_query(query)
    _track_bundle(engine, bundle.bundle_id)
    workspace = str(engine.config.workspace_root)
    return {
        "query": query,
        "workspace_root": workspace,
        "analysis": analysis.model_dump(),
        "bundle": bundle.model_dump(),
        "formatted_context": format_context_message(query, analysis, bundle, workspace),
    }


@mcp.tool(
    name="prepare_context",
    annotations={
        "title": "Prepare budgeted context (analyze + bundle in one call)",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
def prepare_context(
    query: str,
    max_tokens: Optional[int] = None,
    intent: Optional[str] = None,
    workspace_root: Optional[str] = None,
) -> dict:
    """One-call context preparation: analyze intent, fetch a budgeted bundle, return both.

    Prefer this over calling ``analyze_query`` and ``get_context_bundle`` separately.
    The response includes ``formatted_context`` — a ready-to-use markdown block with
    all chunks. Use ``expand_context`` with the returned ``bundle.bundle_id`` only
    if the initial pack is insufficient.
    """
    return _prepare_context(query, max_tokens, intent, workspace_root)


@mcp.prompt(
    name="context",
    title="Context Engineering",
    description="Fetch budgeted codebase context for your task. Usage: /context <question>",
)
def context_prompt(
    query: str = "",
    workspace_root: Optional[str] = None,
) -> str:
    """Slash command: analyze the query and inject a budgeted context pack."""
    task = query.strip() or "Explore this codebase and summarize the main modules."
    result = _prepare_context(task, workspace_root=workspace_root)
    return result["formatted_context"]


@mcp.tool(
    name="analyze_query",
    annotations={
        "title": "Analyze query intent and budget",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
def analyze_query(
    query: str,
    workspace_root: Optional[str] = None,
) -> dict:
    """Classify the query intent and recommend a token budget before retrieval.

    Args:
        query: The user's task or question.
        workspace_root: Project root to index. Defaults to CONTEXT_ENG_WORKSPACE
            env var, then the MCP process cwd (usually the open Cursor project).

    Returns intent, confidence, extracted signals (mentioned files/symbols,
    stack-trace detection), and a recommended/min/max token budget.
    """
    engine = get_engine(workspace_root)
    result = engine.analyze_query(query).model_dump()
    result["workspace_root"] = str(engine.config.workspace_root)
    return result


@mcp.tool(
    name="get_context_bundle",
    annotations={
        "title": "Get a budgeted, query-matched context pack",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
def get_context_bundle(
    query: str,
    max_tokens: Optional[int] = None,
    intent: Optional[str] = None,
    workspace_root: Optional[str] = None,
) -> dict:
    """Return ranked, token-budgeted context chunks for a query.

    Args:
        query: The user's task or question.
        max_tokens: Optional token budget override.
        intent: Optional intent override (debug, implement, explain, refactor, review).
        workspace_root: Project root to index. Defaults to CONTEXT_ENG_WORKSPACE
            env var, then the MCP process cwd (usually the open Cursor project).

    Prefer this over reading whole files. Chunks are symbol slices, import
    neighbors, and keyword snippets packed to fit the budget. Explicitly
    mentioned files/symbols are always included. Use the returned ``bundle_id``
    with ``expand_context`` if you need more.
    """
    engine = get_engine(workspace_root)
    bundle = engine.get_context_bundle(query, max_tokens, intent)
    _track_bundle(engine, bundle.bundle_id)
    result = bundle.model_dump()
    result["workspace_root"] = str(engine.config.workspace_root)
    return result


@mcp.tool(
    name="expand_context",
    annotations={
        "title": "Expand an existing context bundle",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
def expand_context(
    bundle_id: str,
    focus: Optional[str] = None,
    extra_tokens: Optional[int] = None,
) -> dict:
    """Progressively disclose more context for an existing bundle.

    Relaxes filters (extra import hop, optional full file for ``focus``) and
    raises the budget. Call this instead of bulk-reading files when the initial
    bundle was insufficient. ``workspace_root`` is not required here; the bundle
    is looked up by ``bundle_id``.
    """
    engine = _engine_for_bundle(bundle_id)
    if engine is None:
        return {"error": f"unknown bundle_id: {bundle_id}"}
    try:
        bundle = engine.expand_context(bundle_id, focus, extra_tokens)
        result = bundle.model_dump()
        result["workspace_root"] = str(engine.config.workspace_root)
        return result
    except KeyError as exc:
        return {"error": str(exc)}


@mcp.tool(
    name="estimate_tokens",
    annotations={
        "title": "Estimate token count",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
def estimate_tokens(
    text: Optional[str] = None,
    bundle_id: Optional[str] = None,
) -> dict:
    """Estimate tokens for arbitrary text, or for a previously built bundle."""
    if bundle_id:
        engine = _engine_for_bundle(bundle_id)
        if engine is None:
            return {"tokens": 0, "method": "unknown", "error": f"unknown bundle_id: {bundle_id}"}
        return engine.estimate_bundle_tokens(bundle_id).model_dump()
    return get_engine().estimate_tokens(text or "").model_dump()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
