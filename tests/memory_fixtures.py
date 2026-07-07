"""Shared memory-protection test helpers."""

from __future__ import annotations

from seiso.memory.protection import gpu_batch_tier_caps


def mock_gpu_total(monkeypatch, vram_mb: int) -> None:
    monkeypatch.setattr(
        "seiso.memory.protection.discrete_gpu_total_mb",
        lambda _profile=None: vram_mb,
    )


def gpu_normal_caps(vram_mb: int) -> tuple[int, int]:
    return gpu_batch_tier_caps(vram_mb, "normal")
