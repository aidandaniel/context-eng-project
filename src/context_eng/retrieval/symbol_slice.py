"""Extract function/class line ranges so we can send symbol slices, not files.

Python uses the ``ast`` module for accuracy; other languages use a lightweight
brace/indentation heuristic on ``def``/``function``/``class`` declarations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolSpan:
    name: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive


def _python_symbols(source: str) -> list[SymbolSpan]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans: list[SymbolSpan] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            start = node.lineno
            # Include decorators in the span.
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list))
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            spans.append(SymbolSpan(node.name, start, end))
    return spans


_DECL_KEYWORDS = ("function", "def", "class", "func", "fn")


def _brace_symbols(source: str) -> list[SymbolSpan]:
    """Heuristic for brace languages (TS/JS/Java/Go/etc.)."""
    import re

    lines = source.splitlines()
    decl_re = re.compile(
        r"\b(?:export\s+)?(?:async\s+)?"
        r"(?:function|class|func|fn)\s+([A-Za-z_$][\w$]*)"
        r"|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("
        r"|([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{",
    )
    spans: list[SymbolSpan] = []
    for idx, line in enumerate(lines):
        m = decl_re.search(line)
        if not m:
            continue
        name = m.group(1) or m.group(2) or m.group(3)
        if not name:
            continue
        start = idx + 1
        # Walk forward to balance braces starting at this line.
        depth = 0
        seen_open = False
        end = start
        for j in range(idx, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                seen_open = True
            end = j + 1
            if seen_open and depth <= 0:
                break
        spans.append(SymbolSpan(name, start, end))
    return spans


def extract_symbols(source: str, filename: str) -> list[SymbolSpan]:
    if filename.endswith((".py", ".pyi")):
        return _python_symbols(source)
    return _brace_symbols(source)


def find_symbol_span(
    source: str, filename: str, symbol: str
) -> SymbolSpan | None:
    """Return the span for ``symbol`` (case-sensitive), or None."""
    for span in extract_symbols(source, filename):
        if span.name == symbol:
            return span
    # Case-insensitive fallback.
    lowered = symbol.lower()
    for span in extract_symbols(source, filename):
        if span.name.lower() == lowered:
            return span
    return None
