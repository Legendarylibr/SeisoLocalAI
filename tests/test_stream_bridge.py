"""Concurrency coverage for the blocking-to-async stream bridge."""

from __future__ import annotations

import asyncio
import threading

import pytest

from seiso.inference.stream_bridge import (
    StreamBridgeDone,
    ThreadStreamBridge,
)
from seiso.inference.streaming import StreamUpdate


@pytest.mark.asyncio
async def test_stream_bridge_preserves_order_with_backpressure():
    bridge = ThreadStreamBridge(asyncio.get_running_loop(), maxsize=1)
    published: list[bool] = []

    def producer() -> None:
        published.append(bridge.publish(StreamUpdate("a", 1)))
        published.append(bridge.publish(StreamUpdate("b", 2)))
        bridge.producer_finished()

    thread = threading.Thread(target=producer)
    thread.start()
    first = await bridge.next()
    second = await bridge.next()
    done = await bridge.next()
    await bridge.wait_for_producer()
    await asyncio.to_thread(thread.join, 1)

    assert isinstance(first, StreamUpdate) and first.text == "a"
    assert isinstance(second, StreamUpdate) and second.text == "b"
    assert isinstance(done, StreamBridgeDone)
    assert published == [True, True]
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_stream_bridge_cancel_releases_blocked_producer():
    bridge = ThreadStreamBridge(asyncio.get_running_loop(), maxsize=1)
    started = threading.Event()

    def producer() -> None:
        bridge.publish(StreamUpdate("a", 1))
        started.set()
        bridge.publish(StreamUpdate("b", 2))
        bridge.producer_finished()

    thread = threading.Thread(target=producer)
    thread.start()
    await asyncio.to_thread(started.wait, 1)
    bridge.cancel()
    thread.join(timeout=1)

    assert not thread.is_alive()
