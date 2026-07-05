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
        "seiso.memory.platform_profile.performance_headroom_mb", lambda _p: 3072
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
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_GPU_LAYERS"] == "-1"
    assert os.environ["SEISO_LLAMA_BATCH"] == "4096"
    assert os.environ["SEISO_LLAMA_UBATCH"] == "1024"
    assert os.environ["SEISO_LLAMA_CACHE_MB"] == "512"
    assert os.environ.get("SEISO_LLAMA_FLASH_ATTN") == "false"
    assert os.environ.get("SEISO_LLAMA_SPEED_SCALE") == "false"
    assert os.environ["SEISO_STREAM_BATCH_CHARS"] == "16"


def test_platform_profile_linux_nvidia_wsl_uses_non_native_batches(monkeypatch):
    """WSL / non-native Linux NVIDIA keeps flash-attn and headroom-derived batches."""
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
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 24576
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
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: False)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_BATCH"] == "2048"
    assert os.environ["SEISO_LLAMA_UBATCH"] == "512"
    assert os.environ.get("SEISO_LLAMA_FLASH_ATTN") == "true"


def test_platform_profile_linux_nvidia_workstation_uses_conservative_batches(monkeypatch):
    profile = {
        "ram_gb": 128,
        "gpus": [{"name": "NVIDIA RTX 6000 Ada", "vram_total_mb": 49152}],
        "backend": "torch",
        "platform": "Linux",
    }
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 49152
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
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: False)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_BATCH"] == "4096"
    assert os.environ["SEISO_LLAMA_UBATCH"] == "1024"


@pytest.mark.parametrize(
    "name,vram_mb,tier,expected_batch,expected_ubatch",
    [
        ("NVIDIA GeForce GTX 1650", 4096, HardwareTier.EDGE, 512, 256),
        ("NVIDIA GeForce RTX 3050", 6144, HardwareTier.EDGE, 512, 256),
        ("NVIDIA GeForce RTX 3070", 8192, HardwareTier.MODEST, 1024, 256),
        ("NVIDIA GeForce RTX 3060", 12288, HardwareTier.CAPABLE, 1024, 256),
        ("NVIDIA GeForce RTX 4080", 16384, HardwareTier.CAPABLE, 1536, 512),
        ("NVIDIA GeForce RTX 4090", 24576, HardwareTier.WORKSTATION, 4096, 1024),
        ("NVIDIA RTX 6000 Ada", 49152, HardwareTier.WORKSTATION, 4096, 1024),
    ],
)
def test_native_linux_nvidia_batch_caps_all_gpu_tiers(
    monkeypatch, name, vram_mb, tier, expected_batch, expected_ubatch
):
    from seiso.memory.platform_profile import native_linux_nvidia_llama_batch_caps

    batch, ubatch, cache = native_linux_nvidia_llama_batch_caps(
        tier=tier,
        headroom_mb=vram_mb,
        low=False,
    )
    assert batch == expected_batch
    assert ubatch == expected_ubatch
    assert cache == 512


def test_platform_profile_remote_forge_keeps_native_linux_tuning(monkeypatch):
    """allow_remote is a security/bind setting — it must not change memory tuning."""
    profile = {
        "ram_gb": 32,
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24576}],
        "backend": "cuda",
        "platform": "Linux",
    }
    monkeypatch.setenv("SEISO_ALLOW_REMOTE", "1")
    monkeypatch.setenv("SEISO_REMOTE_ACK", "1")
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 1024
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
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    result = apply_platform_memory_profile(profile=profile)

    assert result["memory_profile"] == "balanced"
    assert result["headroom_mb"] == 24576
    assert result["free_headroom_mb"] == 1024
    assert os.environ["SEISO_LLAMA_BATCH"] == "4096"
    assert os.environ["SEISO_LLAMA_UBATCH"] == "1024"
    assert os.environ.get("SEISO_LLAMA_FLASH_ATTN") == "false"


def test_platform_profile_linux_nvidia_modest_sets_safe_batch(monkeypatch):
    profile = {
        "ram_gb": 32,
        "gpus": [{"name": "NVIDIA GeForce RTX 3070", "vram_total_mb": 8192}],
        "backend": "torch",
        "platform": "Linux",
    }
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.MODEST,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 8192
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

    assert os.environ["SEISO_LLAMA_BATCH"] == "1024"
    assert os.environ["SEISO_LLAMA_UBATCH"] == "256"


@pytest.mark.parametrize(
    "name,vram_mb,tier,ram_gb,expected_cache",
    [
        ("NVIDIA GeForce RTX 3070", 8192, HardwareTier.MODEST, 32, "512"),
        ("NVIDIA GeForce RTX 3060", 12288, HardwareTier.CAPABLE, 32, "512"),
        ("NVIDIA GeForce RTX 5080", 16384, HardwareTier.CAPABLE, 48, "512"),
        ("NVIDIA GeForce RTX 4090", 24576, HardwareTier.WORKSTATION, 64, "512"),
        ("NVIDIA RTX 6000 Ada", 49152, HardwareTier.WORKSTATION, 128, "512"),
    ],
)
def test_platform_profile_native_linux_nvidia_all_tiers_are_crash_resistant(
    monkeypatch, name, vram_mb, tier, ram_gb, expected_cache
):
    profile = {
        "ram_gb": ram_gb,
        "gpus": [{"name": name, "vram_total_mb": vram_mb}],
        "backend": "cuda",
        "platform": "Linux",
    }
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: tier,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: vram_mb
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
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_GPU_LAYERS"] == "-1"
    assert int(os.environ["SEISO_LLAMA_BATCH"]) <= 4096
    assert int(os.environ["SEISO_LLAMA_UBATCH"]) <= 1024
    assert os.environ["SEISO_LLAMA_CACHE_MB"] == expected_cache
    assert os.environ["SEISO_LLAMA_FLASH_ATTN"] == "false"
    assert os.environ["SEISO_LLAMA_SPEED_SCALE"] == "false"


def test_platform_profile_workstation_keeps_speed_when_vram_in_use(monkeypatch):
    """Loaded models shrink free VRAM — tuning must use GPU capacity, not free bytes."""
    profile = {
        "ram_gb": 62,
        "gpus": [
            {
                "name": "NVIDIA GeForce RTX 4090",
                "vram_total_mb": 24564,
                "vram_used_mb": 23200,
            }
        ],
        "backend": "cuda",
        "platform": "Linux",
    }
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 1364
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
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    result = apply_platform_memory_profile(profile=profile)

    assert result["memory_profile"] == "balanced"
    assert result["headroom_mb"] == 24564
    assert result["free_headroom_mb"] == 1364
    assert os.environ["SEISO_LLAMA_BATCH"] == "4096"
    assert os.environ["SEISO_LLAMA_UBATCH"] == "1024"
    assert os.environ["SEISO_LLAMA_CACHE_MB"] == "512"


def test_platform_profile_native_linux_clamps_stale_batch_env(monkeypatch):
    profile = {
        "ram_gb": 64,
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24576}],
        "backend": "cuda",
        "platform": "Linux",
    }
    monkeypatch.setenv("SEISO_LLAMA_BATCH", "8192")
    monkeypatch.setenv("SEISO_LLAMA_UBATCH", "2048")
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 24576
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
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_BATCH"] == "4096"
    assert os.environ["SEISO_LLAMA_UBATCH"] == "1024"


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
