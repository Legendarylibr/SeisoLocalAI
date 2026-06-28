"""Tests for hardware privacy sanitization."""

from seiso.security.hardware_privacy import (
    sanitize_env_report,
    sanitize_gpu_record,
    sanitize_gpu_stats,
    sanitize_system_metric_point,
)


def test_sanitize_gpu_record_strips_name():
    raw = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "vram_total_mb": 24576,
        "utilization_pct": 42.0,
    }
    clean = sanitize_gpu_record(raw)
    assert "name" not in clean
    assert clean["index"] == 0
    assert clean["vram_total_mb"] == 24576


def test_sanitize_gpu_stats_list():
    stats = sanitize_gpu_stats(
        [{"index": 0, "name": "Secret GPU", "total_bytes": 1000}]
    )
    assert stats[0]["index"] == 0
    assert "name" not in stats[0]


def test_sanitize_system_metric_point():
    point = {
        "type": "system",
        "cpu_util_pct": 12.0,
        "gpus": [{"index": 0, "name": "RTX 4090", "vram_total_mb": 24000}],
    }
    clean = sanitize_system_metric_point(point)
    assert clean["cpu_util_pct"] == 12.0
    assert "name" not in clean["gpus"][0]


def test_sanitize_env_report_no_device_name():
    raw = {
        "python": "3.11",
        "platform": "Linux-5.15-x86_64",
        "machine": "x86_64",
        "collected_at": "2026-01-01T00:00:00Z",
        "torch": "2.2.0",
        "cuda_available": True,
        "cuda_device": "NVIDIA GeForce RTX 4090",
    }
    clean = sanitize_env_report(raw)
    assert "cuda_device" not in clean
    assert clean["cuda_available"] is True
    assert clean["accelerator_class"] == "discrete_gpu"
