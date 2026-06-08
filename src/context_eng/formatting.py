"""Human-readable formatting for context bundles and analysis."""

from __future__ import annotations

from context_eng.models import Chunk, ContextBundle, QueryAnalysis


def format_context_message(
    query: str,
    analysis: QueryAnalysis,
    bundle: ContextBundle,
    workspace_root: str,
) -> str:
    """Build the prompt body returned by ``/context`` and ``prepare_context``."""
    lines = [
        "# Context Engineering — budgeted codebase context",
        "",
        f"**Query:** {query}",
        f"**Workspace:** {workspace_root}",
        f"**Intent:** {analysis.intent.value} (confidence {analysis.confidence:.0%})",
        (
            f"**Budget:** {bundle.budget_used:,} / {bundle.budget_limit:,} tokens"
            f" ({len(bundle.chunks)} chunks)"
        ),
        f"**Bundle id:** `{bundle.bundle_id}`",
        "",
        "Use only the chunks below to answer or continue the task.",
        "Do not read whole files unless a chunk is clearly insufficient.",
        f"If more context is needed, call `expand_context` with bundle id `{bundle.bundle_id}`.",
        "",
    ]

    if analysis.signals.mentioned_files or analysis.signals.mentioned_symbols:
        anchors = analysis.signals.mentioned_files + analysis.signals.mentioned_symbols
        lines.append(f"**Anchors:** {', '.join(anchors)}")
        lines.append("")

    for i, chunk in enumerate[Chunk](bundle.chunks, start=1):
        lines.extend(
            [
                f"## Chunk {i}: `{chunk.path}` L{chunk.start_line}-L{chunk.end_line}",
                f"*score {chunk.score:.2f} — {chunk.reason} ({chunk.tokens} tokens)*",
                "",
                "```",
                chunk.content.rstrip(),
                "```",
                "",
            ]
        )

    if bundle.excluded_summary:
        lines.extend(["## Excluded", "", bundle.excluded_summary, ""])

    return "\n".join(lines).rstrip() + "\n"
