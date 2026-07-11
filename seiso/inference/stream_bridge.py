"""Bounded bridge from blocking inference producers to async consumers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass

from seiso.inference.streaming import StreamUpdate


@dataclass(frozen=True, slots=True)
class StreamBridgeError:
    exc: BaseException


@dataclass(frozen=True, slots=True)
class StreamBridgeDone:
    pass


StreamBridgeMessage = StreamUpdate | StreamBridgeError | StreamBridgeDone


class ThreadStreamBridge:
    """Apply backpressure and expose cooperative cancellation to a producer."""

    def __init__(self, loop: asyncio.AbstractEventLoop, *, maxsize: int = 32) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[StreamBridgeMessage] = asyncio.Queue(
            maxsize=max(1, maxsize)
        )
        self._cancelled = threading.Event()
        self._producer_done = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def publish(self, message: StreamBridgeMessage) -> bool:
        if self.cancelled:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._queue.put(message),
            self._loop,
        )
        while not self.cancelled:
            try:
                future.result(timeout=0.1)
                return True
            except concurrent.futures.TimeoutError:
                continue
            except (concurrent.futures.CancelledError, RuntimeError):
                return False
        future.cancel()
        return False

    def producer_finished(self) -> None:
        self.publish(StreamBridgeDone())
        self._producer_done.set()

    async def next(self) -> StreamBridgeMessage:
        return await self._queue.get()

    async def wait_for_producer(self, timeout: float = 1.0) -> bool:
        return await asyncio.to_thread(self._producer_done.wait, timeout)
