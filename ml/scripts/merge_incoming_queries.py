"""Merge agent-written YAML shards into the main training corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.query_loader import load_queries

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MAIN = _REPO_ROOT / "ml" / "data" / "budget_training_queries.yaml"
_DEFAULT_INCOMING = _REPO_ROOT / "ml" / "data" / "incoming"


def _format_entry(item: dict) -> str:
    lines = [f"- id: {item['id']}", f'  query: "{item["query"]}"']
    anchors = item.get("expected_anchors", [])
    anchor_parts = ", ".join(f'"{a}"' for a in anchors)
    lines.append(f"  expected_anchors: [{anchor_parts}]")
    if item.get("target_budget") is not None:
        lines.append(f"  target_budget: {int(item['target_budget'])}")
    return "\n".join(lines)


def merge_incoming(
    main_path: Path = _DEFAULT_MAIN,
    incoming_dir: Path = _DEFAULT_INCOMING,
) -> int:
    existing = load_queries(main_path) if main_path.is_file() else []
    seen = {row["id"] for row in existing}

    incoming_paths = sorted(incoming_dir.glob("*.yaml"))
    added = 0
    new_blocks: list[str] = []

    for path in incoming_paths:
        for item in load_queries(path):
            item_id = item["id"]
            if item_id in seen:
                continue
            seen.add(item_id)
            new_blocks.append(_format_entry(item))
            added += 1

    if not new_blocks:
        return 0

    main_path.parent.mkdir(parents=True, exist_ok=True)
    if main_path.is_file():
        text = main_path.read_text(encoding="utf-8").rstrip() + "\n\n"
    else:
        text = (
            "# RF training corpus for budget-bucket labels.\n"
            "# The query text stays natural-language and path-free; expected_anchors are\n"
            "# hidden labels used only for label generation.\n\n"
        )

    text += "\n\n".join(new_blocks) + "\n"
    main_path.write_text(text, encoding="utf-8")

    for path in incoming_paths:
        path.unlink()

    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge incoming training queries.")
    parser.add_argument("--main", type=Path, default=_DEFAULT_MAIN)
    parser.add_argument("--incoming", type=Path, default=_DEFAULT_INCOMING)
    args = parser.parse_args()

    added = merge_incoming(args.main, args.incoming)
    print(f"Merged {added} new queries into {args.main}")


if __name__ == "__main__":
    main()
