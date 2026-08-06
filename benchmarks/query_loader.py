"""Load the benchmark query fixture without requiring PyYAML at runtime."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only when PyYAML is installed
    import yaml as _yaml
except ImportError:  # pragma: no cover - fallback path is what we need here
    _yaml = None


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load the benchmark query list.

    The repo uses a very small YAML subset. When PyYAML is available we defer
    to it; otherwise we parse the file with a narrow fallback that supports the
    current benchmark schema:

    - top-level list items introduced by ``- id:``
    - scalar strings and integers
    - quoted string lists like ``["a", "b"]``
    """
    text = path.read_text(encoding="utf-8")
    if _yaml is not None:
        return _yaml.safe_load(text)
    return _parse_query_yaml(text)


def _parse_query_yaml(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("- "):
            if current is not None:
                rows.append(current)
            current = {}
            line = line[2:].strip()
            if not line:
                continue
            key, value = _split_kv(line)
            current[key] = _parse_value(value)
            continue
        if current is None:
            continue
        if not raw_line.startswith(" "):
            continue
        key, value = _split_kv(line.strip())
        current[key] = _parse_value(value)

    if current is not None:
        rows.append(current)
    return rows


def _split_kv(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"invalid benchmark query line: {line!r}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_value(value: str) -> Any:
    if not value:
        return ""
    if value.startswith("[") or value.startswith("{") or value.startswith(("'", '"')):
        return ast.literal_eval(value)
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    return value
