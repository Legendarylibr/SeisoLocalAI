"""Cross-process serialization for local GPU-heavy work."""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from seiso.env import env_int, env_str

try:  # POSIX: Linux/macOS.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback is best effort.
    fcntl = None  # type: ignore[assignment]


_guard = threading.RLock()
_handle = None
_depth = 0


def _lock_path() -> Path:
    raw = env_str(
        "SEISO_GPU_RESOURCE_LOCK_PATH",
        env_str("SEISO_INFERENCE_LOCK_PATH", ""),
    ).strip()
    if raw:
        return Path(raw).expanduser()
    data_dir = Path(env_str("SEISO_DATA_DIR", "~/.seiso")).expanduser()
    return data_dir / "locks" / "gpu-resource.lock"


def _timeout_s() -> float:
    return max(
        1,
        env_int(
            "SEISO_GPU_RESOURCE_LOCK_TIMEOUT_S",
            env_int("SEISO_INFERENCE_PROCESS_LOCK_TIMEOUT_S", 600),
        ),
    )


def _acquire_file_lock(timeout_s: float):
    path = _lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "seiso-gpu-resource.lock"
        handle = fallback.open("a+")

    if fcntl is None:
        return handle

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise TimeoutError(
                    "Timed out waiting for another Seiso GPU task to finish"
                ) from None
            time.sleep(0.1)


def acquire_gpu_resource_lock() -> None:
    """Acquire the process-wide GPU resource lock, reentrant in this process."""
    global _depth, _handle
    with _guard:
        if _depth == 0:
            _handle = _acquire_file_lock(_timeout_s())
        _depth += 1


def release_gpu_resource_lock() -> None:
    """Release one level of the process-wide GPU resource lock."""
    global _depth, _handle
    with _guard:
        if _depth <= 0:
            return
        if _depth > 1:
            _depth -= 1
            return
        _depth = 0
        handle = _handle
        _handle = None
        if handle is None:
            return
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def gpu_resource_lock_held_by_other_process() -> bool:
    """Return True when another process owns the shared GPU lock."""
    with _guard:
        if _depth > 0:
            return False
    if fcntl is None:
        return False
    path = _lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "seiso-gpu-resource.lock"
        handle = fallback.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


@contextmanager
def gpu_resource_lock() -> Iterator[None]:
    acquire_gpu_resource_lock()
    try:
        yield
    finally:
        release_gpu_resource_lock()
