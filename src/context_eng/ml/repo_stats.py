"""Cached repo size features for RF budget prediction (H2).

Avoids reading every workspace file on each ``extract_features`` call by
preferring manifest metadata (line counts already computed at index time).
"""

from __future__ import annotations

import math
from pathlib import Path

from context_eng.cache import TtlLruCache
from context_eng.config import Config
from context_eng.workspace import iter_files, read_text

# Bounded so long-lived MCP processes cannot grow this without limit.
_STATS_CACHE: TtlLruCache[tuple[str, int], tuple[int, float]] = TtlLruCache(maxsize=32)


def clear_repo_stats_cache() -> None:
    """Drop process-local stats cache (tests / forced refresh)."""
    _STATS_CACHE.clear()


def _walk_repo_stats(config: Config) -> tuple[int, float]:
    """Fallback: walk workspace and count lines by reading files."""
    workspace = Path(config.workspace_root).resolve()
    file_count = 0
    total_lines = 0
    for path in iter_files(workspace, config.ignore_globs):
        file_count += 1
        try:
            text = read_text(path)
        except OSError:
            continue
        total_lines += text.count("\n") + (1 if text else 0)
    return file_count, math.log10(total_lines + 1)


def repo_stats(config: Config) -> tuple[int, float]:
    """Return ``(file_count, log10(loc+1))`` using manifest stats when possible."""
    workspace = Path(config.workspace_root).resolve()
    key_root = workspace.as_posix()

    if config.manifest_auto_build:
        from context_eng.index.manifest import (
            get_manifest,
            manifest_path,
            repo_stats_from_manifest,
        )

        try:
            mpath = manifest_path(workspace)
            mtime_ns = mpath.stat().st_mtime_ns if mpath.is_file() else -1
        except OSError:
            mtime_ns = -1

        cache_key = (key_root, mtime_ns)
        cached = _STATS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            stats = repo_stats_from_manifest(get_manifest(workspace, config))
        except OSError:
            stats = _walk_repo_stats(config)
        else:
            # Refresh key if manifest was (re)written during get_manifest.
            try:
                mpath = manifest_path(workspace)
                if mpath.is_file():
                    cache_key = (key_root, mpath.stat().st_mtime_ns)
            except OSError:
                pass
        _STATS_CACHE[cache_key] = stats
        return stats

    return _walk_repo_stats(config)
