"""Pure-Python keyword retriever with snippet extraction.

Uses ripgrep when available for speed, otherwise a Python scan (the MVP target
machine has no ``rg``, so the fallback is the primary path and is fully tested).
Candidate ``keyword_match`` is a raw hit count; normalization happens in the
ranker so all feature weighting lives in one place.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from context_eng.config import Config
from context_eng.index.manifest import get_searchable_files
from context_eng.models import CandidateChunk
from context_eng.workspace import read_text, relpath

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "at", "is", "are", "was", "were", "be", "been", "this",
    "that", "it", "with", "from", "by", "as", "we", "i", "you", "my", "our",
    "fix", "add", "make", "use", "using", "how", "does", "do", "what", "why",
    "when", "where", "should", "would", "could", "please", "can", "after",
    "before", "into", "out", "up", "down", "not", "no", "yes",
}

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def extract_keywords(query: str, extra: list[str] | None = None) -> list[str]:
    """Derive search keywords from a query plus optional extra terms."""
    raw = _WORD_RE.findall(query)
    for term in extra or []:
        raw.extend(_WORD_RE.findall(term))

    keywords: list[str] = []
    for word in raw:
        lw = word.lower()
        if lw in _STOPWORDS:
            continue
        keywords.append(word)
        # Also include camelCase sub-tokens so "refreshToken" matches "refresh".
        parts = _CAMEL_SPLIT_RE.split(word)
        if len(parts) > 1:
            keywords.extend(p for p in parts if len(p) >= 3)

    # Dedupe case-insensitively, preserve order.
    seen: dict[str, str] = {}
    for kw in keywords:
        seen.setdefault(kw.lower(), kw)
    return list(seen.values())


def _merge_line_groups(
    hit_lines: list[int], context: int, total_lines: int
) -> list[tuple[int, int]]:
    """Merge nearby hit lines into (start, end) 1-based inclusive ranges."""
    if not hit_lines:
        return []
    hit_lines = sorted(set(hit_lines))
    groups: list[tuple[int, int]] = []
    start = max(1, hit_lines[0] - context)
    end = min(total_lines, hit_lines[0] + context)
    for ln in hit_lines[1:]:
        new_start = max(1, ln - context)
        if new_start <= end + 1:
            end = min(total_lines, ln + context)
        else:
            groups.append((start, end))
            start = new_start
            end = min(total_lines, ln + context)
    groups.append((start, end))
    return groups


def rg_available() -> bool:
    """Return True when the ``rg`` binary is on PATH."""
    return shutil.which("rg") is not None


def _python_file_hits(
    path: Path, patterns: list[tuple[str, re.Pattern[str]]]
) -> list[int]:
    source = read_text(path)
    if not source:
        return []
    hit_lines: list[int] = []
    for idx, line in enumerate(source.splitlines()):
        if any(rx.search(line) for _, rx in patterns):
            hit_lines.append(idx + 1)
    return hit_lines


def _python_scan_hits(
    files: list[Path],
    patterns: list[tuple[str, re.Pattern[str]]],
    workspace: Path,
) -> dict[str, list[int]]:
    hits: dict[str, list[int]] = {}
    for path in files:
        hit_lines = _python_file_hits(path, patterns)
        if hit_lines:
            hits[relpath(path, workspace)] = hit_lines
    return hits


def _parse_rg_json(stdout: str, workspace: Path) -> dict[str, list[int]]:
    workspace = workspace.resolve()
    hits: dict[str, list[int]] = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "match":
            continue
        data = payload.get("data") or {}
        raw_path = str((data.get("path") or {}).get("text") or "")
        line_no = int(data.get("line_number") or 0)
        if not raw_path or line_no <= 0:
            continue
        rel = relpath(Path(raw_path), workspace)
        hits.setdefault(rel, []).append(line_no)
    return hits


def _ripgrep_file_hits(
    files: list[Path],
    keywords: list[str],
    workspace: Path,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[dict[str, list[int]], bool]:
    """Run ripgrep once via ``--files-from``.

    Returns ``(hits, ok)``. ``ok`` is False on timeout, spawn failure, or
    unexpected exit codes so callers can fall back to a Python scan.
    """
    if not files or not keywords:
        return {}, True
    list_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", delete=False
        ) as fh:
            for path in files:
                fh.write(f"{path.resolve()}\n")
            list_path = Path(fh.name)
        args = [
            "rg",
            "--json",
            "--line-number",
            "--no-heading",
            "--fixed-strings",
            "--files-from",
            str(list_path),
        ]
        for kw in keywords:
            args.extend(["-e", kw])
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
            cwd=str(workspace.resolve()),
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}, False
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)

    if proc.returncode not in (0, 1):
        return {}, False
    return _parse_rg_json(proc.stdout, workspace), True


def _chunks_from_hits(
    *,
    rel_path: str,
    hit_lines: list[int],
    workspace: Path,
    patterns: list[tuple[str, re.Pattern[str]]],
    context: int,
) -> list[CandidateChunk]:
    path = workspace / rel_path
    source = read_text(path)
    if not source:
        return []
    lines = source.splitlines()
    chunks: list[CandidateChunk] = []
    for start, end in _merge_line_groups(hit_lines, context, len(lines)):
        snippet = "\n".join(lines[start - 1 : end])
        distinct = sum(1 for _, rx in patterns if rx.search(snippet))
        chunks.append(
            CandidateChunk(
                path=rel_path,
                start_line=start,
                end_line=end,
                content=snippet,
                tier="grep",
                keyword_match=float(distinct),
            )
        )
    return chunks


class GrepRetriever:
    """Keyword retriever implementing the Retriever protocol."""

    def __init__(self, config: Config):
        self.config = config
        self.last_degraded = False

    def search(
        self, query: str, workspace: Path, limit: int
    ) -> list[CandidateChunk]:
        self.last_degraded = False
        keywords = extract_keywords(query)
        if not keywords:
            return []
        patterns = [(kw, re.compile(re.escape(kw), re.IGNORECASE)) for kw in keywords]
        context = self.config.grep_context_lines
        workspace = workspace.resolve()
        files = get_searchable_files(workspace, self.config)
        timeout = float(getattr(self.config, "rg_timeout_seconds", 30.0))

        hit_map: dict[str, list[int]] = {}
        if rg_available():
            hit_map, ok = _ripgrep_file_hits(
                files, keywords, workspace, timeout_seconds=timeout
            )
            if not ok:
                self.last_degraded = True
                hit_map = _python_scan_hits(files, patterns, workspace)
        else:
            hit_map = _python_scan_hits(files, patterns, workspace)

        candidates: list[CandidateChunk] = []
        for rel, hit_lines in hit_map.items():
            candidates.extend(
                _chunks_from_hits(
                    rel_path=rel,
                    hit_lines=hit_lines,
                    workspace=workspace,
                    patterns=patterns,
                    context=context,
                )
            )

        candidates.sort(key=lambda c: (-c.keyword_match, c.path, c.start_line))
        return candidates[:limit]
