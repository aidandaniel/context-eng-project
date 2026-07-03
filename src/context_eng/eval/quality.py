"""Quality metrics for MCP bundles — eval/labeling only (not runtime)."""

from __future__ import annotations

from typing import Any, Protocol


class _BundleLike(Protocol):
    chunks: list[Any]


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _path_present(chunk_paths: set[str], target: str) -> bool:
    normalized = _normalize_path(target)
    return any(p == normalized or p.endswith(normalized) for p in chunk_paths)


def relevant_file_recall(expected_files: list[str], bundle_paths: list[str]) -> float:
    """Fraction of labeled relevant files represented in a bundle's paths."""
    if not expected_files:
        return 1.0
    present = {_normalize_path(p) for p in bundle_paths}
    hits = sum(1 for path in expected_files if _path_present(present, path))
    return hits / len(expected_files)


def task_rubric_pass(bundle: _BundleLike, rubric: dict[str, Any]) -> bool:
    """Return whether a bundle satisfies eval-only task rubric rules.

    Supported rubric keys:
    - ``expected_paths``: every path must appear in the bundle
    - ``any_of_paths``: for each group, at least one path must appear
    - ``content_contains``: each substring must appear in combined chunk text
    """
    if not rubric:
        return True

    chunk_paths = {_normalize_path(c.path) for c in bundle.chunks}
    combined = "\n".join(getattr(c, "content", "") for c in bundle.chunks)

    for path in rubric.get("expected_paths") or []:
        if not _path_present(chunk_paths, str(path)):
            return False

    groups = rubric.get("any_of_paths") or []
    if groups:
        matched_group = False
        for group in groups:
            if not isinstance(group, list):
                continue
            if any(_path_present(chunk_paths, str(path)) for path in group):
                matched_group = True
                break
        if not matched_group:
            return False

    for needle in rubric.get("content_contains") or []:
        if str(needle) not in combined:
            return False

    return True
