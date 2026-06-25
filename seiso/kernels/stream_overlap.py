"""CUDA stream pool for overlapping independent QKV / MLP projections."""

from __future__ import annotations

import contextlib
from typing import Iterator

_STREAMS: list | None = None
_NUM_STREAMS = 3  # Q, K, V


def _ensure_streams():
    global _STREAMS
    if _STREAMS is not None:
        return _STREAMS
    try:
        import torch

        if not torch.cuda.is_available():
            _STREAMS = []
            return _STREAMS
        _STREAMS = [torch.cuda.Stream() for _ in range(_NUM_STREAMS)]
    except ImportError:
        _STREAMS = []
    return _STREAMS


def stream_for_index(index: int):
    """Return a dedicated CUDA stream for parallel projection overlap."""
    streams = _ensure_streams()
    if not streams:
        import torch

        return torch.cuda.current_stream()
    return streams[index % len(streams)]


@contextlib.contextmanager
def overlap_stream(index: int) -> Iterator[None]:
    """Context manager: run body on stream `index`, sync back to default."""
    streams = _ensure_streams()
    if not streams:
        yield
        return
    import torch

    stream = streams[index % len(streams)]
    default = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        yield
    default.wait_stream(stream)


def sync_overlap_streams() -> None:
    """Wait for all overlap streams before the next optimizer step."""
    streams = _ensure_streams()
    if not streams:
        return
    import torch

    default = torch.cuda.current_stream()
    for s in streams:
        default.wait_stream(s)


def release_overlap_streams() -> None:
    global _STREAMS
    _STREAMS = None