"""Report corpus bucket deficits for the Ralph migration loop."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TARGETS = _REPO_ROOT / "ml" / "data" / "corpus_targets.yaml"
_DEFAULT_QUERIES = _REPO_ROOT / "ml" / "data" / "budget_training_queries.yaml"
_DEFAULT_LABELS = _REPO_ROOT / "ml" / "data" / "budget_labels.jsonl"

_ID_RE = re.compile(r"^\s*-\s+id:", re.MULTILINE)
_Y_RE_TEMPLATE = r'"y":\s*{bucket}(?:,|\}})'


def _load_targets(path: Path) -> dict[str, Any]:
    if yaml is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    text = path.read_text(encoding="utf-8")
    targets: dict[str, Any] = {"min_total": 180, "min_per_bucket": 20, "batch_size": 10}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("min_total:"):
            targets["min_total"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("min_per_bucket:"):
            targets["min_per_bucket"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("batch_size:"):
            targets["batch_size"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("- ") and stripped[2:].strip().isdigit():
            targets.setdefault("buckets", []).append(int(stripped[2:].strip()))
    if "buckets" not in targets:
        targets["buckets"] = [
            2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000,
        ]
    return targets


def _query_count(queries_path: Path) -> int:
    if not queries_path.is_file():
        return 0
    return len(_ID_RE.findall(queries_path.read_text(encoding="utf-8")))


def _bucket_counts(labels_path: Path, buckets: list[int]) -> dict[int, int]:
    counts = {bucket: 0 for bucket in buckets}
    if not labels_path.is_file():
        return counts
    text = labels_path.read_text(encoding="utf-8")
    for bucket in buckets:
        counts[bucket] = len(re.findall(_Y_RE_TEMPLATE.format(bucket=bucket), text))
    return counts


def _has_expected_tokens(labels_path: Path) -> bool:
    if not labels_path.is_file():
        return False
    return '"expected_tokens"' in labels_path.read_text(encoding="utf-8")


def compute_gap(
    *,
    targets_path: Path = _DEFAULT_TARGETS,
    queries_path: Path = _DEFAULT_QUERIES,
    labels_path: Path = _DEFAULT_LABELS,
) -> dict[str, Any]:
    targets = _load_targets(targets_path)
    min_total = int(targets["min_total"])
    min_per_bucket = int(targets["min_per_bucket"])
    batch_size = int(targets.get("batch_size", 10))
    buckets = [int(b) for b in targets["buckets"]]

    query_count = _query_count(queries_path)
    bucket_counts = _bucket_counts(labels_path, buckets)

    remaining = max(0, min_total - query_count)
    deficits: dict[int, int] = {}
    for bucket in buckets:
        deficit = max(0, min_per_bucket - bucket_counts[bucket])
        if deficit:
            deficits[bucket] = deficit
            remaining += deficit

    if not _has_expected_tokens(labels_path):
        remaining = max(remaining, 1)

    primary_bucket = min(deficits, key=deficits.get) if deficits else buckets[0]
    deficit_buckets = ", ".join(str(b) for b in sorted(deficits))

    anchor_guidance = "2-4" if primary_bucket <= 4000 else "4-8"
    if primary_bucket >= 10000:
        anchor_guidance = "8-20"

    return {
        "remaining": remaining,
        "query_count": query_count,
        "min_total": min_total,
        "min_per_bucket": min_per_bucket,
        "batch_size": batch_size,
        "deficits": deficits,
        "primary_bucket": primary_bucket,
        "deficit_buckets": deficit_buckets or str(primary_bucket),
        "anchor_guidance": anchor_guidance,
        "timestamp": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report ML corpus gaps.")
    parser.add_argument("--targets", type=Path, default=_DEFAULT_TARGETS)
    parser.add_argument("--queries", type=Path, default=_DEFAULT_QUERIES)
    parser.add_argument("--labels", type=Path, default=_DEFAULT_LABELS)
    parser.add_argument("--export-shell", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    gap = compute_gap(
        targets_path=args.targets,
        queries_path=args.queries,
        labels_path=args.labels,
    )

    if args.export_shell:
        exports = {
            "DEFICIT_BUCKETS": gap["deficit_buckets"],
            "BATCH_SIZE": str(gap["batch_size"]),
            "PRIMARY_BUCKET": str(gap["primary_bucket"]),
            "ANCHOR_GUIDANCE": gap["anchor_guidance"],
            "TIMESTAMP": gap["timestamp"],
            "REMAINING": str(gap["remaining"]),
        }
        for key, value in exports.items():
            safe = value.replace('"', '\\"')
            print(f'export {key}="{safe}"')
        return

    if args.json:
        print(json.dumps(gap, sort_keys=True))
        return

    print(f"remaining={gap['remaining']}")
    if gap["deficits"]:
        for bucket, deficit in sorted(gap["deficits"].items()):
            print(f"  bucket {bucket}: need {deficit} more labeled rows")


if __name__ == "__main__":
    main()
