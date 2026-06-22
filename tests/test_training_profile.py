"""CUDA training profile selection tests."""

from __future__ import annotations


def test_resolve_cuda_training_mode_tiers():
    from seiso.kernels.training_profile import CudaTrainingMode, resolve_cuda_training_mode

    assert resolve_cuda_training_mode(headroom_mb=4096) == CudaTrainingMode.LEAN
    assert resolve_cuda_training_mode(headroom_mb=12000) == CudaTrainingMode.BALANCED
    assert resolve_cuda_training_mode(headroom_mb=24000, est_train_mb=4000) == CudaTrainingMode.SPEED


def test_prepare_profile_speed_disables_checkpointing(monkeypatch):
    from seiso.kernels.training_profile import prepare_cuda_training_profile

    monkeypatch.setenv("SEISO_KERNEL_AUTO_TUNE", "0")
    monkeypatch.setattr(
        "seiso.kernels.training_profile.native_cuda_kernels_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "seiso.training.platform_caps.training_capabilities",
        lambda: {"fused_kernels_available": True},
    )

    profile = prepare_cuda_training_profile(
        headroom_mb=24576,
        est_train_mb=6000,
        model_id="meta-llama/Llama-3.1-8B",
        batch_size=2,
        max_seq_length=2048,
    )
    assert profile["cuda_training_mode"] == "speed"
    assert profile["gradient_checkpointing"] is False
    assert profile["use_fused_ce"] is True
    assert profile["use_fused_lora"] is True


def test_prepare_profile_lean_enables_checkpointing(monkeypatch):
    from seiso.kernels.training_profile import prepare_cuda_training_profile

    monkeypatch.setenv("SEISO_KERNEL_AUTO_TUNE", "0")
    monkeypatch.setattr(
        "seiso.kernels.training_profile.native_cuda_kernels_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "seiso.training.platform_caps.training_capabilities",
        lambda: {"fused_kernels_available": True},
    )

    profile = prepare_cuda_training_profile(
        headroom_mb=4096,
        est_train_mb=9000,
        model_id="meta-llama/Llama-3.1-8B",
        batch_size=1,
        max_seq_length=2048,
    )
    assert profile["cuda_training_mode"] == "lean"
    assert profile["gradient_checkpointing"] is True
    assert profile["kernel_low_vram"] is True
    import os

    assert os.environ.get("SEISO_KERNEL_LOW_VRAM") == "1"


def test_guess_hidden_dim():
    from seiso.kernels.training_profile import guess_hidden_dim

    assert guess_hidden_dim("meta-llama/Llama-3.1-70B") == 8192
    assert guess_hidden_dim("Qwen/Qwen2.5-3B") == 2048
    assert guess_hidden_dim("meta-llama/Llama-3.1-8B") == 4096
