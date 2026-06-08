"""Application settings for the fixture app."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    issuer: str = "fixture-app"
    access_ttl_seconds: int = 900
    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
