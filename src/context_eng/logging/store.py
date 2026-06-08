"""Append-only JSONL event logger.

One line per request. Fields are intentionally flat and stable so they can be
loaded directly into a dataframe to train a budget model later. ``success``
stays null in the MVP (manual/offline labeling).
"""

from __future__ import annotations

import json
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
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            # Logging must never break context retrieval.
            pass
