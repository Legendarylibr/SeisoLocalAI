"""Progress reporting helpers for Hugging Face downloads."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]

_EMIT_INTERVAL_S = 0.35
_MIN_BYTES_DELTA = 256 * 1024  # 256 KiB — skip noisy updates on fast links


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
            self._last_emit_bytes = 0
            self._speed_anchor = (time.monotonic(), 0)

        def _rolling_speed_bps(self, now: float, nbytes: int) -> float:
            t0, n0 = self._speed_anchor
            dt = max(now - t0, 0.001)
            if dt >= 3.0 or nbytes <= n0:
                self._speed_anchor = (now, nbytes)
                elapsed = max(now - self._started_at, 0.001)
                return max(0.0, nbytes / elapsed)
            return max(0.0, (nbytes - n0) / dt)

        def update(self, n=1):
            ret = super().update(n)
            if not callback or not self.total:
                return ret
            now = time.monotonic()
            nbytes = int(self.n)
            finished = nbytes >= self.total
            byte_delta = abs(nbytes - self._last_emit_bytes)
            if (
                not finished
                and now - self._last_emit < _EMIT_INTERVAL_S
                and byte_delta < _MIN_BYTES_DELTA
            ):
                return ret
            self._last_emit = now
            self._last_emit_bytes = nbytes
            rate = self._rolling_speed_bps(now, nbytes)
            remaining_bytes = max(self.total - nbytes, 0)
            eta_seconds = int(remaining_bytes / rate) if rate > 0 else None
            callback(
                {
                    "phase": "download",
                    "bytes": nbytes,
                    "total_bytes": int(self.total),
                    "percent": round(100.0 * nbytes / self.total, 2),
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
