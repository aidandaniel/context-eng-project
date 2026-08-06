"""Bounded in-memory caches with LRU eviction and optional TTL."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(slots=True)
class _Entry(Generic[V]):
    value: V
    expires_at: float | None


class TtlLruCache(Generic[K, V]):
    """OrderedDict-backed cache: max size + optional idle/absolute TTL.

    - ``get`` / ``__getitem__`` refresh LRU order and (when ``ttl_seconds`` is
      set) reset the expiry clock (idle TTL).
    - Inserts that exceed ``maxsize`` evict the least-recently-used key.
    - Expired entries are dropped on access or when space is needed.
    """

    def __init__(
        self,
        maxsize: int,
        ttl_seconds: float | None = None,
        *,
        on_evict: Callable[[K, V], None] | None = None,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0 when set")
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._on_evict = on_evict
        self._data: OrderedDict[K, _Entry[V]] = OrderedDict()

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, Hashable):
            return False
        return self.get(key) is not None  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[K]:
        self._purge_expired()
        return iter(self._data)

    def clear(self) -> None:
        if self._on_evict is not None:
            for key, entry in list(self._data.items()):
                self._on_evict(key, entry.value)
        self._data.clear()

    def get(self, key: K, default: V | None = None) -> V | None:
        entry = self._data.get(key)
        if entry is None:
            return default
        if self._expired(entry):
            self._pop(key)
            return default
        self._data.move_to_end(key)
        entry.expires_at = self._fresh_expiry()
        return entry.value

    def __getitem__(self, key: K) -> V:
        missing = object()
        value = self.get(key, default=missing)  # type: ignore[arg-type]
        if value is missing:
            raise KeyError(key)
        return value  # type: ignore[return-value]

    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)

    def set(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = _Entry(value=value, expires_at=self._fresh_expiry())
            return
        self._purge_expired()
        while len(self._data) >= self.maxsize:
            old_key, old_entry = self._data.popitem(last=False)
            if self._on_evict is not None:
                self._on_evict(old_key, old_entry.value)
        self._data[key] = _Entry(value=value, expires_at=self._fresh_expiry())

    def pop(self, key: K, default: V | None = None) -> V | None:
        entry = self._data.pop(key, None)
        if entry is None:
            return default
        if self._on_evict is not None:
            self._on_evict(key, entry.value)
        return entry.value

    def keys(self) -> list[K]:
        self._purge_expired()
        return list(self._data.keys())

    def _fresh_expiry(self) -> float | None:
        if self.ttl_seconds is None:
            return None
        return monotonic() + self.ttl_seconds

    def _expired(self, entry: _Entry[V]) -> bool:
        return entry.expires_at is not None and monotonic() >= entry.expires_at

    def _purge_expired(self) -> None:
        if self.ttl_seconds is None:
            return
        now = monotonic()
        for key in list(self._data):
            entry = self._data[key]
            if entry.expires_at is not None and now >= entry.expires_at:
                self._pop(key)

    def _pop(self, key: K) -> None:
        entry = self._data.pop(key, None)
        if entry is not None and self._on_evict is not None:
            self._on_evict(key, entry.value)
