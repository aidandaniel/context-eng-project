"""CLI to warm the workspace file manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from context_eng.config import load_config
from context_eng.index.manifest import build_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build workspace search manifest.")
    parser.add_argument("--workspace", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(str(args.workspace) if args.workspace else None)
    manifest = build_manifest(cfg.workspace_root, cfg)
    path = save_manifest(manifest, cfg.workspace_root)
    print(f"Indexed {manifest.file_count} files -> {path}")


if __name__ == "__main__":
    main()
