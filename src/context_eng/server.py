"""MCPServer exposing the Context Engineering tools.

The MCP layer is intentionally thin: it validates inputs and delegates to
``ContextEngine``. Run with ``python -m context_eng.server`` (stdio transport).

Multi-project support: pass ``workspace_root`` on each call (recommended when
using a global MCP config), or rely on ``CONTEXT_ENG_WORKSPACE`` / process cwd.
Engines are cached per workspace (LRU); bundles are tracked globally by
``bundle_id`` (LRU + idle TTL) so ``expand_context`` works without re-passing
the workspace without unbounded memory growth.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from mcp.server.mcpserver import MCPServer

from context_eng import __version__
from context_eng.cache import TtlLruCache
from context_eng.config import load_config
from context_eng.engine import ContextEngine
from context_eng.formatting import format_context_message
from context_eng.ml.model_paths import DEFAULT_MODEL_NAME
from context_eng.workspace_resolve import resolve_workspace

mcp = MCPServer("context-eng")

# Process-wide cache bounds (H1). Overridable via env for long-running hosts.
_MAX_CACHED_ENGINES = max(1, int(os.environ.get("CONTEXT_ENG_MAX_ENGINES", "8")))
_MAX_BUNDLE_OWNERS = max(1, int(os.environ.get("CONTEXT_ENG_MAX_BUNDLE_OWNERS", "64")))
_BUNDLE_OWNER_TTL_SECONDS = float(
    os.environ.get("CONTEXT_ENG_BUNDLE_TTL_SECONDS", "1800")
)

# bundle_id -> engine that created it (for expand_context / estimate_tokens).
_bundle_owners: TtlLruCache[str, ContextEngine] = TtlLruCache(
    maxsize=_MAX_BUNDLE_OWNERS,
    ttl_seconds=_BUNDLE_OWNER_TTL_SECONDS if _BUNDLE_OWNER_TTL_SECONDS > 0 else None,
)


def _on_engine_evict(_key: str, engine: ContextEngine) -> None:
    """Drop bundle ownership rows that pointed at an evicted engine."""
    for bundle_id in list(_bundle_owners.keys()):
        if _bundle_owners.get(bundle_id) is engine:
            _bundle_owners.pop(bundle_id, None)


# One engine per resolved workspace path (posix key).
_engines: TtlLruCache[str, ContextEngine] = TtlLruCache(
    maxsize=_MAX_CACHED_ENGINES,
    on_evict=_on_engine_evict,
)

def get_engine(workspace_root: str | None = None) -> ContextEngine:
    """Return a cached ContextEngine for ``workspace_root`` (or cwd/env default)."""
    root = resolve_workspace(workspace_root)
    key = root.as_posix()
    engine = _engines.get(key)
    if engine is None:
        engine = ContextEngine(config=load_config(str(root)))
        _engines[key] = engine
    return engine


def _engine_for_bundle(bundle_id: str) -> ContextEngine | None:
    return _bundle_owners.get(bundle_id)


def _track_bundle(engine: ContextEngine, bundle_id: str) -> None:
    _bundle_owners[bundle_id] = engine


def _tool_error(exc: BaseException) -> dict:
    """Structured error payload for MCP tools (never re-raise from entrypoints)."""
    return {"error": str(exc), "error_type": type(exc).__name__}


def _attach_meta(engine: ContextEngine, result: dict) -> dict:
    """Add workspace, version, and config_error without clobbering tool fields."""
    result.setdefault("workspace_root", str(engine.config.workspace_root))
    result["version"] = __version__
    if engine.config.config_error:
        result["config_error"] = engine.config.config_error
    return result


# Explicit names kept for clarity; Exception covers the rest.
_TOOL_EXCEPTIONS = (
    PermissionError,
    FileNotFoundError,
    NotADirectoryError,
    ValueError,
    OSError,
    Exception,
)


def _prepare_context(
    query: str,
    max_tokens: Optional[int] = None,
    intent: Optional[str] = None,
    workspace_root: Optional[str] = None,
) -> dict:
    """Analyze a query and return a ready-to-use context bundle."""
    try:
        engine = get_engine(workspace_root)
        bundle = engine.get_context_bundle(query, max_tokens, intent)
        analysis = engine.analysis_for_bundle(bundle.bundle_id) or engine.analyze_query(query)
        _track_bundle(engine, bundle.bundle_id)
        workspace = str(engine.config.workspace_root)
        return _attach_meta(
            engine,
            {
                "query": query,
                "workspace_root": workspace,
                "analysis": analysis.model_dump(),
                "bundle": bundle.model_dump(),
                "formatted_context": format_context_message(
                    query, analysis, bundle, workspace
                ),
            },
        )
    except _TOOL_EXCEPTIONS as exc:
        return _tool_error(exc)


@mcp.tool(
    name="prepare_context",
    annotations={
        "title": "Prepare budgeted context (analyze + bundle in one call)",
        "readOnlyHint": False,
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
    try:
        return _prepare_context(query, max_tokens, intent, workspace_root)
    except _TOOL_EXCEPTIONS as exc:
        return _tool_error(exc)


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
    try:
        task = query.strip() or "Explore this codebase and summarize the main modules."
        result = _prepare_context(task, workspace_root=workspace_root)
        if "error" in result:
            return f"Error preparing context: {result['error']}"
        return result["formatted_context"]
    except _TOOL_EXCEPTIONS as exc:
        return f"Error preparing context: {exc}"


@mcp.tool(
    name="analyze_query",
    annotations={
        "title": "Analyze query intent and budget",
        "readOnlyHint": False,
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
    try:
        engine = get_engine(workspace_root)
        result = engine.analyze_query(query).model_dump()
        return _attach_meta(engine, result)
    except _TOOL_EXCEPTIONS as exc:
        return _tool_error(exc)


@mcp.tool(
    name="get_context_bundle",
    annotations={
        "title": "Get a budgeted, query-matched context pack",
        "readOnlyHint": False,
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
    try:
        engine = get_engine(workspace_root)
        bundle = engine.get_context_bundle(query, max_tokens, intent)
        _track_bundle(engine, bundle.bundle_id)
        result = bundle.model_dump()
        return _attach_meta(engine, result)
    except _TOOL_EXCEPTIONS as exc:
        return _tool_error(exc)


@mcp.tool(
    name="expand_context",
    annotations={
        "title": "Expand an existing context bundle",
        "readOnlyHint": False,
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
    try:
        engine = _engine_for_bundle(bundle_id)
        if engine is None:
            return {"error": f"unknown bundle_id: {bundle_id}"}
        bundle = engine.expand_context(bundle_id, focus, extra_tokens)
        result = bundle.model_dump()
        return _attach_meta(engine, result)
    except _TOOL_EXCEPTIONS as exc:
        return _tool_error(exc)


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
    try:
        if bundle_id:
            engine = _engine_for_bundle(bundle_id)
            if engine is None:
                return {
                    "tokens": 0,
                    "method": "unknown",
                    "error": f"unknown bundle_id: {bundle_id}",
                }
            return engine.estimate_bundle_tokens(bundle_id).model_dump()
        return get_engine().estimate_tokens(text or "").model_dump()
    except _TOOL_EXCEPTIONS as exc:
        return _tool_error(exc)


@mcp.tool(
    name="mcp_health",
    annotations={
        "title": "Context-eng health",
        "readOnlyHint": True,
        "openWorldHint": False,
    },
)
def mcp_health() -> dict:
    """Process liveness, package version, and default RF model name."""
    try:
        return {
            "ok": True,
            "version": __version__,
            "model": DEFAULT_MODEL_NAME,
            "python": sys.version.split()[0],
        }
    except _TOOL_EXCEPTIONS as exc:
        return _tool_error(exc)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
