"""Dedicated thread pools for Forge blocking work.

GPU-bound jobs share a single worker so CUDA contexts stay serialized.
I/O-bound jobs (recipes import, hub publish prep) use a small pool.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forge-gpu")
IO_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="forge-io")


def shutdown_executors(*, wait: bool = False) -> None:
    GPU_EXECUTOR.shutdown(wait=wait, cancel_futures=False)
    IO_EXECUTOR.shutdown(wait=wait, cancel_futures=False)
