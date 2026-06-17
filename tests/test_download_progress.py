"""Tests for Hugging Face download progress reporting."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from forge.services.download_progress import make_tqdm_class


def test_progress_tqdm_emits_bytes_and_rolling_speed():
    events: list[dict] = []
    ProgressTqdm = make_tqdm_class(events.append)

    bar = ProgressTqdm(total=2_000_000_000)
    bar.update(500_000_000)
    time.sleep(0.05)
    bar.update(500_000_000)

    assert len(events) >= 1
    last = events[-1]
    assert last["phase"] == "download"
    assert last["bytes"] == 1_000_000_000
    assert last["total_bytes"] == 2_000_000_000
    assert last["percent"] == 50.0
    assert last["speed_bps"] > 0


def test_progress_tqdm_emits_on_completion():
    callback = MagicMock()
    ProgressTqdm = make_tqdm_class(callback)

    bar = ProgressTqdm(total=1024)
    bar.update(1024)

    callback.assert_called()
    payload = callback.call_args[0][0]
    assert payload["bytes"] == 1024
    assert payload["percent"] == 100.0


def test_progress_tqdm_throttles_small_updates():
    events: list[dict] = []
    ProgressTqdm = make_tqdm_class(events.append)

    bar = ProgressTqdm(total=10_000_000)
    bar.update(1024)
    bar.update(1024)
    assert len(events) <= 1
