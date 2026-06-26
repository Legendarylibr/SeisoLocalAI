"""Low-VRAM kernel mode tests."""

from __future__ import annotations


def test_kernel_low_vram_env_override(monkeypatch):
    from seiso.kernels.memory_mode import kernel_low_vram_enabled

    monkeypatch.setenv("SEISO_KERNEL_LOW_VRAM", "1")
    assert kernel_low_vram_enabled() is True
    monkeypatch.setenv("SEISO_KERNEL_LOW_VRAM", "0")
    assert kernel_low_vram_enabled() is False


def test_kernel_low_vram_from_headroom(monkeypatch):
    from seiso.kernels.memory_mode import kernel_low_vram_enabled

    monkeypatch.delenv("SEISO_KERNEL_LOW_VRAM", raising=False)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 4096)
    assert kernel_low_vram_enabled() is True
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    assert kernel_low_vram_enabled() is False


def test_estimate_vram_savings_low_vram_bonus():
    from seiso.kernels.dispatch import estimate_vram_savings_pct

    base = estimate_vram_savings_pct(True, False, low_vram=False)
    low = estimate_vram_savings_pct(True, False, low_vram=True)
    assert low >= base


def test_training_memory_guards_set_low_vram_env(monkeypatch):
    from seiso.memory.protection import apply_training_memory_guards
    from seiso.training import platform_caps
    from seiso.training.config import TrainConfig

    platform_caps.training_capabilities.cache_clear()
    monkeypatch.delenv("SEISO_KERNEL_LOW_VRAM", raising=False)
    monkeypatch.setenv("SEISO_KERNEL_AUTO_TUNE", "0")
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 4096)
    monkeypatch.setattr(
        "seiso.memory.protection.hardware_profile",
        lambda: {"backend": "torch", "gpus": [{"vram_total_mb": 4096}], "ram_gb": 16},
    )
    monkeypatch.setattr(
        "seiso.memory.protection.training_defaults",
        lambda _p: {
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 2048,
        },
    )
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda *_a, **_k: 12000,
    )
    monkeypatch.setattr(
        "seiso.training.platform_caps.training_capabilities",
        lambda: {
            "fused_kernels_available": True,
            "fused_ce_available": True,
            "fused_lora_available": False,
            "recommended_quant": "4bit",
            "supports_qlora": True,
            "kernel_backend": "cuda",
            "train_platform": "cuda",
            "multi_gpu_available": False,
        },
    )
    monkeypatch.setattr(
        "seiso.kernels.training_profile.native_cuda_kernels_available",
        lambda: False,
    )

    cfg = apply_training_memory_guards(
        TrainConfig(
            model_id="test/model",
            dataset="data.jsonl",
            batch_size=2,
            max_seq_length=2048,
            use_fused_ce=False,
            gradient_checkpointing=False,
        )
    )
    import os

    assert os.environ.get("SEISO_KERNEL_LOW_VRAM") == "1"
    assert cfg.gradient_checkpointing is True
    # Explicit use_fused_ce=False is respected; tight VRAM also keeps fused CE off.
    assert cfg.use_fused_ce is False
    assert cfg.max_seq_length <= 1024
