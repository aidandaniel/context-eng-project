"""Local embedding retriever (optional ``sentence-transformers`` extra).

Loads models from the local Hugging Face cache only (``local_files_only``).
When the extra is not installed or the model is missing, ``search`` returns [].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from context_eng.config import Config
from context_eng.models import CandidateChunk
from context_eng.workspace import iter_files, read_text, relpath

_CHUNK_LINES = 24
_CHUNK_STRIDE = 12
_MAX_FILES = 80


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _line_windows(
    total_lines: int, window: int, stride: int
) -> list[tuple[int, int]]:
    if total_lines <= 0:
        return []
    if total_lines <= window:
        return [(1, total_lines)]
    windows: list[tuple[int, int]] = []
    start = 1
    while start <= total_lines:
        end = min(total_lines, start + window - 1)
        windows.append((start, end))
        if end >= total_lines:
            break
        start += stride
    return windows


class EmbeddingRetriever:
    """Semantic retriever implementing the Retriever protocol."""

    def __init__(self, config: Config):
        self.config = config
        self._model: Any | None = None
        self._model_failed = False

    def _load_model(self) -> Any | None:
        if self._model_failed:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._model_failed = True
            return None
        try:
            self._model = SentenceTransformer(
                self.config.embedding_model_name,
                local_files_only=True,
            )
        except (OSError, ValueError, RuntimeError):
            self._model_failed = True
            self._model = None
        return self._model

    def search(
        self, query: str, workspace: Path, limit: int
    ) -> list[CandidateChunk]:
        if limit <= 0 or not query.strip():
            return []
        model = self._load_model()
        if model is None:
            return []

        windows: list[tuple[str, int, int, str]] = []
        file_count = 0
        for path in iter_files(workspace, self.config.ignore_globs):
            if file_count >= _MAX_FILES:
                break
            source = read_text(path)
            if not source:
                continue
            file_count += 1
            rel = relpath(path, workspace)
            lines = source.splitlines()
            for start, end in _line_windows(len(lines), _CHUNK_LINES, _CHUNK_STRIDE):
                snippet = "\n".join(lines[start - 1 : end])
                if snippet.strip():
                    windows.append((rel, start, end, snippet))

        if not windows:
            return []

        texts = [query] + [w[3] for w in windows]
        vectors = model.encode(texts, normalize_embeddings=True)
        query_vec = vectors[0]
        scored: list[tuple[float, CandidateChunk]] = []
        for idx, (rel, start, end, snippet) in enumerate(windows, start=1):
            sim = _cosine_similarity(list(query_vec), list(vectors[idx]))
            scored.append(
                (
                    sim,
                    CandidateChunk(
                        path=rel,
                        start_line=start,
                        end_line=end,
                        content=snippet,
                        tier="embedding",
                        keyword_match=sim,
                    ),
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]
