"""Tests for OS/tier memory profile defaults."""

from __future__ import annotations

import os

import pytest

from seiso.hardware.tiers import HardwareTier
from seiso.memory.platform_profile import (
    apply_platform_memory_profile,
    memory_profile_label,
)


@pytest.fixture(autouse=True)
def _clear_llama_env(monkeypatch):
    for key in list(os.environ):
        if (
            key.startswith("SEISO_LLAMA_")
            or key == "SEISO_MEMORY_PROFILE"
            or key == "SEISO_SKIP_MLX_PROBE"
        ):
            monkeypatch.delenv(key, raising=False)


def test_memory_profile_label_low_on_tight_headroom():
    assert memory_profile_label({"ram_gb": 12, "gpus": []}) == "low"


def test_memory_profile_label_balanced_on_roomy_machine():
    assert (
        memory_profile_label({"ram_gb": 64, "gpus": [{"vram_total_mb": 49152}]})
        == "balanced"
    )


def test_platform_profile_darwin_16gb_apple(monkeypatch):
    profile = {"ram_gb": 16, "gpus": [], "backend": "metal", "platform": "Darwin"}
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.APPLE_UNIFIED,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 3072
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {
            "supports_mlx_inference": False,
            "gpu_count": 0,
            "train_platform": "cpu",
        },
    )
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    result = apply_platform_memory_profile(profile=profile)

    assert result["memory_profile"] == "low"
    assert os.environ["SEISO_LLAMA_USE_MMAP"] == "true"
    assert os.environ["SEISO_LLAMA_USE_MLOCK"] == "false"
    assert os.environ["SEISO_LLAMA_PROMPT_CACHE"] == "true"
    assert os.environ["SEISO_LLAMA_CACHE_MB"] == "1024"
    assert os.environ.get("SEISO_SKIP_MLX_PROBE") == "1"


def test_platform_profile_disable_memory_caps_keeps_full_defaults(monkeypatch):
    profile = {"ram_gb": 16, "gpus": [], "backend": "metal", "platform": "Darwin"}
    monkeypatch.setenv("SEISO_DISABLE_MEMORY_CAPS", "1")
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.APPLE_UNIFIED,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 3072
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {
            "supports_mlx_inference": True,
            "gpu_count": 0,
            "train_platform": "cpu",
        },
    )
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_PROMPT_CACHE"] == "true"
    assert os.environ["SEISO_LLAMA_CACHE_MB"] == "1024"
    assert "SEISO_LLAMA_BATCH" not in os.environ


def test_platform_profile_darwin_intel_cpu_only(monkeypatch):
    profile = {"ram_gb": 16, "gpus": [], "backend": "cpu", "platform": "Darwin"}
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier", lambda _p: HardwareTier.CPU_ONLY
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 8192
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {
            "supports_mlx_inference": False,
            "gpu_count": 0,
            "train_platform": "cpu",
        },
    )
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("os.cpu_count", lambda: 8)

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_GPU_LAYERS"] == "0"
    assert os.environ["SEISO_LLAMA_THREADS"] == "6"


def test_platform_profile_windows_no_cuda(monkeypatch):
    profile = {"ram_gb": 16, "gpus": [], "backend": "cpu", "platform": "Windows"}
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier", lambda _p: HardwareTier.CPU_ONLY
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 8192
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {"gpu_count": 0, "train_platform": "cpu"},
    )
    monkeypatch.setattr("platform.system", lambda: "Windows")

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_GPU_LAYERS"] == "0"


def test_platform_profile_linux_nvidia_uses_gpu_layers(monkeypatch):
    profile = {
        "ram_gb": 64,
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24576}],
        "backend": "torch",
        "platform": "Linux",
    }
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 20480
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {
            "gpu_count": 1,
            "train_platform": "cpu",
            "nvidia_hardware": True,
            "vendor": "nvidia",
        },
    )
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_GPU_LAYERS"] == "-1"
    assert os.environ["SEISO_LLAMA_BATCH"] == "2048"
    assert os.environ["SEISO_LLAMA_UBATCH"] == "512"
    assert os.environ["SEISO_LLAMA_CACHE_MB"] == "2048"
    assert os.environ["SEISO_STREAM_BATCH_CHARS"] == "16"


def test_apply_only_setdefault(monkeypatch):
    profile = {"ram_gb": 16, "gpus": [], "backend": "cpu", "platform": "Darwin"}
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier", lambda _p: HardwareTier.CPU_ONLY
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 8192
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {"gpu_count": 0, "train_platform": "cpu"},
    )
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    os.environ["SEISO_LLAMA_BATCH"] = "1024"

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_BATCH"] == "1024"


def test_assess_catalog_fit_moe_uses_download_bytes():
    from seiso.hardware.fit import assess_catalog_fit

    model = {
        "params": "35B",
        "quant": "Q4_K_M",
        "tags": ("moe",),
        "repo_id": "org/MoE",
        "download_bytes": 20 * 1024**3,
        "task": "chat",
    }
    profile = {"ram_gb": 32, "gpus": [{"vram_total_mb": 24576}], "backend": "metal"}

    fit = assess_catalog_fit(model, profile)
    assert fit["est_vram_mb"] >= 20 * 1024
