"""Build a labeled JSONL dataset from SWE-bench Lite oracle / BM25 packs.

Example:
  python scripts/build_swebench_budget_dataset.py --out ml/data/swebench_lite_budget.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running without editable install.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from context_eng.ml.swebench_labels import load_swebench_lite_examples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "ml" / "data" / "swebench_lite_budget.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--split", default="test", help="HF split (default: test)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of instances (for smoke runs)",
    )
    args = parser.parse_args()

    examples = load_swebench_lite_examples(split=args.split, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_record(), ensure_ascii=False) + "\n")

    buckets = Counter(ex.label_budget for ex in examples)
    print(f"Wrote {len(examples)} examples -> {args.out}")
    print("label_budget counts:", dict(sorted(buckets.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
