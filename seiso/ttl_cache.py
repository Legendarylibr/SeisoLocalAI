"""Thread-safe TTL cache with bounded size for Hub and API memoization."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TtlCache(Generic[K, V]):
    """In-memory cache with per-entry TTL and LRU-style eviction at max capacity."""

    def __init__(self, *, ttl_s: float, max_entries: int = 512) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max(1, max_entries)
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            ts, value = entry
            if now - ts >= self._ttl_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: K, value: V) -> None:
        now = time.monotonic()
        with self._lock:
            if key not in self._data and len(self._data) >= self._max_entries:
                self._purge_expired(now)
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _purge_expired(self, now: float) -> None:
        expired = [
            key for key, (ts, _) in self._data.items() if now - ts >= self._ttl_s
        ]
        for key in expired:
            del self._data[key]
