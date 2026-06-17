"""Progress reporting helpers for Hugging Face downloads."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]


def make_tqdm_class(callback: ProgressCallback | None):
    """Build a tqdm subclass that forwards byte progress to callback."""
    from tqdm import tqdm

    class ProgressTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("unit", "B")
            kwargs.setdefault("unit_scale", True)
            kwargs.setdefault("unit_divisor", 1024)
            super().__init__(*args, **kwargs)
            self._started_at = time.monotonic()
            self._last_emit = 0.0

        def update(self, n=1):
            ret = super().update(n)
            if not callback or not self.total:
                return ret
            now = time.monotonic()
            if now - self._last_emit < 0.15 and self.n < self.total:
                return ret
            self._last_emit = now
            elapsed = max(now - self._started_at, 0.001)
            rate = self.n / elapsed
            remaining_bytes = max(self.total - self.n, 0)
            eta_seconds = int(remaining_bytes / rate) if rate > 0 else None
            callback(
                {
                    "phase": "download",
                    "bytes": int(self.n),
                    "total_bytes": int(self.total),
                    "percent": round(100.0 * self.n / self.total, 1),
                    "eta_seconds": eta_seconds,
                    "speed_bps": rate,
                }
            )
            return ret

    return ProgressTqdm


def estimate_load_eta_seconds(size_bytes: int) -> int:
    """Rough ETA for loading a model into VRAM from local disk."""
    if size_bytes <= 0:
        return 8
    # ~150 MB/s effective for mmap + weight init on typical hardware
    return max(5, int(size_bytes / (150 * 1024 * 1024)) + 3)
