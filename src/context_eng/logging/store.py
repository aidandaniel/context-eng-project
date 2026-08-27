"""Append-only JSONL event logger.

One line per request. Fields are intentionally flat and stable so they can be
loaded directly into a dataframe to train a budget model later. ``success``
stays null in the MVP (manual/offline labeling).

JSON is also written to stderr so a local MCP host can collect logs without
relying on workspace disk.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, events_path: Path):
        self.events_path = events_path

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def log(self, event: dict[str, Any]) -> None:
        """Append a single event; never raise into the request path."""
        record = {"timestamp": time.time(), **event}
        line = json.dumps(record, default=str)
        try:
            print(line, file=sys.stderr, flush=True)
        except OSError:
            pass
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            # Disk logging must never break context retrieval.
            pass
