"""Canonical GPU enumeration tests."""

from seiso.hardware.gpus import (
    clear_gpu_enumeration_cache,
    enumerate_compute_gpus,
    enumerate_gpus,
    gpu_count,
    primary_gpu_name,
)


def test_enumerate_compute_gpus_skips_mlx(monkeypatch):
    clear_gpu_enumeration_cache()
    monkeypatch.setattr("seiso.hardware.gpus._torch_gpus", lambda: [])
    monkeypatch.setattr("seiso.hardware.gpus._nvidia_smi_gpus", lambda: [])
    monkeypatch.setattr(
        "seiso.hardware.gpus._mlx_apple_gpu",
        lambda: [
            {
                "index": 0,
                "name": "Apple GPU (MLX)",
                "vram_total_mb": None,
                "vram_used_mb": None,
                "utilization_pct": None,
                "temperature_c": None,
            }
        ],
    )
    clear_gpu_enumeration_cache()

    assert len(enumerate_gpus()) == 1
    assert enumerate_gpus()[0]["name"] == "Apple GPU (MLX)"
    clear_gpu_enumeration_cache()
    assert enumerate_compute_gpus() == ()
    clear_gpu_enumeration_cache()


def test_gpu_count_and_primary_name(monkeypatch):
    clear_gpu_enumeration_cache()
    monkeypatch.setattr(
        "seiso.hardware.gpus._torch_gpus",
        lambda: [
            {
                "index": 0,
                "name": "NVIDIA RTX",
                "vram_total_mb": 8192,
                "vram_used_mb": None,
                "utilization_pct": None,
                "temperature_c": None,
            }
        ],
    )
    monkeypatch.setattr("seiso.hardware.gpus._nvidia_smi_gpus", lambda: [])
    monkeypatch.setattr("seiso.hardware.gpus._mlx_apple_gpu", lambda: [])
    clear_gpu_enumeration_cache()

    assert gpu_count() == 1
    assert primary_gpu_name() == "NVIDIA RTX"
    clear_gpu_enumeration_cache()
