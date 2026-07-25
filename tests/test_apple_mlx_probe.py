"""Apple MLX GPU probe — report full unified memory_size, not Metal soft budget."""

from __future__ import annotations

import sys
import types

from seiso.hardware.probes import apple as apple_probe


def test_probe_apple_mlx_gpu_uses_memory_size_not_recommended(monkeypatch):
    monkeypatch.delenv("SEISO_SKIP_MLX_PROBE", raising=False)

    fake_core = types.ModuleType("mlx.core")

    def device_info() -> dict:
        return {
            "device_name": "Apple M4 Pro",
            # Soft Metal budget — must NOT be used as capacity.
            "max_recommended_working_set_size": 19_069_665_280,
            "memory_size": 25_769_803_776,  # 24 GiB
        }

    def get_active_memory() -> int:
        return 512 * 1024 * 1024

    fake_metal = types.ModuleType("mlx.core.metal")
    fake_metal.get_active_memory = get_active_memory  # type: ignore[attr-defined]
    fake_core.device_info = device_info  # type: ignore[attr-defined]
    fake_core.get_active_memory = get_active_memory  # type: ignore[attr-defined]
    fake_core.metal = fake_metal  # type: ignore[attr-defined]

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    gpus = apple_probe.probe_apple_mlx_gpu()
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu["name"] == "Apple M4 Pro (MLX)"
    assert gpu["unified_memory"] is True
    assert gpu["vram_total_mb"] == 24576.0
    assert gpu["vram_used_mb"] == 512.0


def test_probe_apple_mlx_skip_env(monkeypatch):
    monkeypatch.setenv("SEISO_SKIP_MLX_PROBE", "1")
    assert apple_probe.probe_apple_mlx_gpu() == []
