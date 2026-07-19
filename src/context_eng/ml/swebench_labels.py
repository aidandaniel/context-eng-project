"""Derive budget-bucket labels from SWE-bench Lite oracle / BM25 packs.

Uses Hugging Face packs:

- ``SWE-bench/SWE-bench_Lite`` (queries + gold patches)
- ``princeton-nlp/SWE-bench_Lite_oracle``
- ``princeton-nlp/SWE-bench_Lite_bm25_13K``
- ``princeton-nlp/SWE-bench_Lite_bm25_27K``

Labels are the smallest ``BUDGET_BUCKETS`` ceiling that fits the oracle
``<code>`` block (gold files). BM25 packs supply retrieval size proxies for
features when a full repo checkout is unavailable.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from context_eng.config import Config
from context_eng.intent.classifier import analyze
from context_eng.ml.budget_model import BUDGET_BUCKETS, snap_to_bucket
from context_eng.ml.features import FEATURE_NAMES, extract_features
from context_eng.tokens import count_tokens

LITE_DATASET = "SWE-bench/SWE-bench_Lite"
ORACLE_DATASET = "princeton-nlp/SWE-bench_Lite_oracle"
BM25_13K_DATASET = "princeton-nlp/SWE-bench_Lite_bm25_13K"
BM25_27K_DATASET = "princeton-nlp/SWE-bench_Lite_bm25_27K"

_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
_FILE_START_RE = re.compile(r"\[start of ([^\]]+)\]")
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/", re.MULTILINE)


@dataclass(frozen=True)
class SwebenchBudgetExample:
    """One supervised training row for the budget RF."""

    instance_id: str
    repo: str
    query: str
    label_budget: int
    label_source: str
    oracle_tokens: int
    bm25_13k_tokens: int
    bm25_27k_tokens: int
    oracle_files: tuple[str, ...]
    patch_files: tuple[str, ...]
    features: dict[str, float | int]

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["oracle_files"] = list(self.oracle_files)
        record["patch_files"] = list(self.patch_files)
        return record


def extract_code_block(text: str) -> str:
    """Return the ``<code>...</code>`` body from a SWE-bench prompt, if present."""
    match = _CODE_RE.search(text or "")
    return match.group(1) if match else ""


def files_in_code_block(code: str) -> list[str]:
    """Ordered unique paths marked with ``[start of path]`` in a code pack."""
    return list(dict.fromkeys(_FILE_START_RE.findall(code or "")))


def files_in_patch(patch: str) -> list[str]:
    """Paths touched by a unified diff ``patch`` field."""
    return list(dict.fromkeys(_DIFF_FILE_RE.findall(patch or "")))


def context_token_count(text: str) -> int:
    """Token count of the retrieved/oracle code pack (not the full prompt)."""
    code = extract_code_block(text)
    return count_tokens(code) if code else count_tokens(text or "")


def label_budget_from_oracle_tokens(oracle_tokens: int) -> int:
    """Map oracle context size onto the runtime budget bucket grid."""
    if oracle_tokens <= 0:
        return BUDGET_BUCKETS[0]
    return snap_to_bucket(oracle_tokens)


def _repo_proxies(bm25_27k_tokens: int, pack_file_count: int) -> tuple[int, float]:
    """Cheap repo-size proxies when the workspace is not checked out."""
    # BM25-27K packs are truncated retrieval surfaces, not full repos; scale up
    # file count so the feature stays in a plausible repo range.
    file_count = max(pack_file_count * 25, pack_file_count, 1)
    # Rough LOC from tokens (~1 token ≈ 0.75 line of code on average for source).
    loc_estimate = max(int(bm25_27k_tokens * 0.75), pack_file_count, 1)
    return file_count, math.log10(loc_estimate + 1)


def build_example(
    *,
    instance_id: str,
    repo: str,
    problem_statement: str,
    patch: str,
    oracle_text: str,
    bm25_13k_text: str,
    bm25_27k_text: str,
    config: Config | None = None,
) -> SwebenchBudgetExample:
    """Build features + oracle-derived budget label for one SWE-bench instance."""
    cfg = config or Config(workspace_root=Path(".").resolve())
    query = problem_statement or ""
    analysis = analyze(query, cfg)

    oracle_tokens = context_token_count(oracle_text)
    bm25_13k_tokens = context_token_count(bm25_13k_text)
    bm25_27k_tokens = context_token_count(bm25_27k_text)
    oracle_files = files_in_code_block(extract_code_block(oracle_text))
    patch_files = files_in_patch(patch)
    pack_files = files_in_code_block(extract_code_block(bm25_27k_text))
    repo_file_count, repo_loc_log = _repo_proxies(bm25_27k_tokens, len(pack_files))

    # Prefer gold oracle file count; fall back to patch paths.
    anchor_count = len(oracle_files) or len(patch_files)
    # Do not set must_include to oracle_tokens — that leaks the label into X.
    # Use patch size + a coarse per-anchor prior (runtime uses real anchor fits).
    patch_tokens = count_tokens(patch or "")
    must_include_estimate = max(patch_tokens, anchor_count * 1000)
    features = extract_features(
        query,
        analysis,
        cfg,
        discovered_anchor_count=anchor_count,
        must_include_token_estimate=must_include_estimate,
        repo_file_count=repo_file_count,
        repo_loc_log=repo_loc_log,
    )
    # Fold pack breadth into retrieval_signal_count without changing schema.
    features["retrieval_signal_count"] = int(features["retrieval_signal_count"]) + len(
        pack_files
    )

    return SwebenchBudgetExample(
        instance_id=instance_id,
        repo=repo,
        query=query,
        label_budget=label_budget_from_oracle_tokens(oracle_tokens),
        label_source="swebench_lite_oracle_tokens",
        oracle_tokens=oracle_tokens,
        bm25_13k_tokens=bm25_13k_tokens,
        bm25_27k_tokens=bm25_27k_tokens,
        oracle_files=tuple(oracle_files),
        patch_files=tuple(patch_files),
        features=features,
    )


def _index_by_instance(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["instance_id"]): row for row in rows if row.get("instance_id")}


def build_examples_from_splits(
    lite_rows: Iterable[Mapping[str, Any]],
    oracle_rows: Iterable[Mapping[str, Any]],
    bm25_13k_rows: Iterable[Mapping[str, Any]],
    bm25_27k_rows: Iterable[Mapping[str, Any]],
    *,
    config: Config | None = None,
    limit: int | None = None,
) -> list[SwebenchBudgetExample]:
    """Join Lite + oracle + BM25 splits on ``instance_id`` and emit examples."""
    oracle = _index_by_instance(oracle_rows)
    bm13 = _index_by_instance(bm25_13k_rows)
    bm27 = _index_by_instance(bm25_27k_rows)

    examples: list[SwebenchBudgetExample] = []
    for row in lite_rows:
        instance_id = str(row.get("instance_id") or "")
        if not instance_id:
            continue
        if instance_id not in oracle or instance_id not in bm13 or instance_id not in bm27:
            continue
        examples.append(
            build_example(
                instance_id=instance_id,
                repo=str(row.get("repo") or oracle[instance_id].get("repo") or ""),
                problem_statement=str(row.get("problem_statement") or ""),
                patch=str(row.get("patch") or ""),
                oracle_text=str(oracle[instance_id].get("text") or ""),
                bm25_13k_text=str(bm13[instance_id].get("text") or ""),
                bm25_27k_text=str(bm27[instance_id].get("text") or ""),
                config=config,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def load_swebench_lite_examples(
    *,
    split: str = "test",
    limit: int | None = None,
    config: Config | None = None,
) -> list[SwebenchBudgetExample]:
    """Download Hugging Face SWE-bench Lite packs and build labeled examples."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "SWE-bench labeling requires datasets: pip install -e '.[swebench]'"
        ) from exc

    lite = load_dataset(LITE_DATASET, split=split)
    oracle = load_dataset(ORACLE_DATASET, split=split)
    bm13 = load_dataset(BM25_13K_DATASET, split=split)
    bm27 = load_dataset(BM25_27K_DATASET, split=split)
    return build_examples_from_splits(
        lite,
        oracle,
        bm13,
        bm27,
        config=config,
        limit=limit,
    )


def feature_matrix(
    examples: Iterable[SwebenchBudgetExample],
) -> tuple[list[list[float]], list[int], list[str]]:
    """Return (X, y, feature_names) for sklearn training."""
    x_rows: list[list[float]] = []
    y: list[int] = []
    for example in examples:
        x_rows.append([float(example.features[name]) for name in FEATURE_NAMES])
        y.append(int(example.label_budget))
    return x_rows, y, list(FEATURE_NAMES)
