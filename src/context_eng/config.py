"""Configuration for the Context Engineering MCP server.

Values can be overridden by a `context-eng.toml` file at the workspace root.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from context_eng.workspace_resolve import resolve_workspace

DEFAULT_IGNORE_GLOBS: tuple[str, ...] = (
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".context-eng",
)

# Intent -> (recommended, min, max) token budgets.
DEFAULT_INTENT_BUDGETS: dict[str, tuple[int, int, int]] = {
    "debug": (6000, 3000, 9000),
    "implement": (8000, 4000, 12000),
    "explain": (4000, 2000, 6000),
    "refactor": (10000, 5000, 15000),
    "review": (5000, 2500, 8000),
}


@dataclass(frozen=True)
class Config:
    workspace_root: Path
    ignore_globs: tuple[str, ...] = DEFAULT_IGNORE_GLOBS
    default_max_tokens: int = 8000
    intent_budgets: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_INTENT_BUDGETS)
    )
    grep_context_lines: int = 8
    max_grep_candidates: int = 50
    enable_anchor_inference: bool = True
    max_inferred_anchor_files: int = 3
    inferred_anchor_min_score: float = 1.0
    # Optional (non-anchor) chunks below this normalized score are dropped even
    # if budget remains -- the budget is a ceiling, not a fill target.
    min_chunk_score: float = 0.15
    # Optional chunk cap: None uses adaptive_max_optional_chunks per query.
    max_optional_chunks: int | None = None
    max_optional_chunks_upper: int = 4
    max_optional_chunks_floor: int = 1
    events_path: Path | None = None
    # ``rf`` uses ``budget_rf_v2.joblib`` (default); ``intent`` is legacy and ignored at runtime.
    budget_source: str = "rf"
    ml_model_path: Path | None = None
    enable_embedding_retriever: bool = False
    embedding_model_name: str = "all-MiniLM-L6-v2"

    @property
    def resolved_events_path(self) -> Path:
        if self.events_path is not None:
            return self.events_path
        return self.workspace_root / ".context-eng" / "events.jsonl"


def load_config(workspace_root: str | None = None) -> Config:
    """Build a Config, layering optional `context-eng.toml` over defaults."""

    root = resolve_workspace(workspace_root)
    cfg = Config(workspace_root=root)

    toml_path = root / "context-eng.toml"
    if not toml_path.is_file():
        return cfg

    try:
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return cfg

    section = data.get("context_eng", data)
    overrides: dict[str, object] = {}

    if "ignore_globs" in section:
        overrides["ignore_globs"] = tuple(section["ignore_globs"])
    if "default_max_tokens" in section:
        overrides["default_max_tokens"] = int(section["default_max_tokens"])
    if "grep_context_lines" in section:
        overrides["grep_context_lines"] = int(section["grep_context_lines"])
    if "max_grep_candidates" in section:
        overrides["max_grep_candidates"] = int(section["max_grep_candidates"])
    if "enable_anchor_inference" in section:
        overrides["enable_anchor_inference"] = bool(section["enable_anchor_inference"])
    if "max_inferred_anchor_files" in section:
        overrides["max_inferred_anchor_files"] = int(section["max_inferred_anchor_files"])
    if "inferred_anchor_min_score" in section:
        overrides["inferred_anchor_min_score"] = float(section["inferred_anchor_min_score"])
    if "min_chunk_score" in section:
        overrides["min_chunk_score"] = float(section["min_chunk_score"])
    if "max_optional_chunks" in section:
        overrides["max_optional_chunks"] = int(section["max_optional_chunks"])
    if "max_optional_chunks_upper" in section:
        overrides["max_optional_chunks_upper"] = int(section["max_optional_chunks_upper"])
    if "max_optional_chunks_floor" in section:
        overrides["max_optional_chunks_floor"] = int(section["max_optional_chunks_floor"])
    if "budget_source" in section:
        overrides["budget_source"] = str(section["budget_source"])
    if "ml_model_path" in section:
        overrides["ml_model_path"] = Path(section["ml_model_path"])
    if "enable_embedding_retriever" in section:
        overrides["enable_embedding_retriever"] = bool(section["enable_embedding_retriever"])
    if "embedding_model_name" in section:
        overrides["embedding_model_name"] = str(section["embedding_model_name"])
    if "intent_budgets" in section:
        budgets = dict(DEFAULT_INTENT_BUDGETS)
        for intent, vals in section["intent_budgets"].items():
            budgets[intent] = (int(vals[0]), int(vals[1]), int(vals[2]))
        overrides["intent_budgets"] = budgets

    return replace(cfg, **overrides)
