"""Ignore-pattern matching for workspace traversal (H3).

Patterns support ``fnmatch`` globs (e.g. ``*.pem``, ``**/vendor/**``) while
keeping bare names like ``.git`` / ``node_modules`` as directory-segment matches
for backward compatibility. Secret basenames stay in ``is_secret_path``.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

_GLOB_META = set("*?[]")


def _has_glob_meta(pattern: str) -> bool:
    return any(ch in _GLOB_META for ch in pattern)


def path_is_ignored(rel_posix: str, ignore_globs: tuple[str, ...]) -> bool:
    """Return True if a workspace-relative POSIX path matches an ignore pattern."""
    if not rel_posix:
        return False
    rel = rel_posix.replace("\\", "/")
    parts = PurePosixPath(rel).parts
    name = PurePosixPath(rel).name
    for pattern in ignore_globs:
        pat = pattern.replace("\\", "/").strip()
        if not pat:
            continue
        if not _has_glob_meta(pat):
            # Bare token: match a path segment or the full basename.
            if pat in parts or pat == name or pat == rel:
                return True
            continue
        if fnmatch.fnmatchcase(rel, pat) or fnmatch.fnmatchcase(name, pat):
            return True
        if not pat.startswith("**/") and fnmatch.fnmatchcase(rel, f"**/{pat}"):
            return True
        for part in parts:
            if fnmatch.fnmatchcase(part, pat):
                return True
    return False


def merge_ignore_globs(
    defaults: tuple[str, ...],
    extra: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Union ``extra`` onto ``defaults`` (order-preserving, no duplicates)."""
    merged: list[str] = list(defaults)
    seen = set(defaults)
    for item in extra:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return tuple(merged)
