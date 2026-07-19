"""Logging helpers shared across the fixture app."""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class _Logger:
    name: str

    def _emit(self, level: str, msg: str, *args: object) -> None:
        formatted = msg % args if args else msg
        print(f"[{level}] {self.name}: {formatted}", file=sys.stderr)

    def info(self, msg: str, *args: object) -> None:
        self._emit("INFO", msg, *args)

    def warning(self, msg: str, *args: object) -> None:
        self._emit("WARN", msg, *args)

    def error(self, msg: str, *args: object) -> None:
        self._emit("ERROR", msg, *args)


def get_logger(name: str) -> _Logger:
    return _Logger(name=name)
