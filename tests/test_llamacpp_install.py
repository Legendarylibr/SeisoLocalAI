"""Tests for shared llama-cpp-python install logic."""

from __future__ import annotations


def test_pip_install_strategies_cuda_first_when_preferred():
    from seiso.inference.llamacpp_install import pip_install_strategies

    strategies = pip_install_strategies(prefer_cuda=True)
    assert strategies[0][-1] == "https://abetlen.github.io/llama-cpp-python/whl/cu124"
    assert any("cu121" in " ".join(cmd) for cmd in strategies)


def test_pip_install_strategies_cpu_only_when_not_preferred():
    from seiso.inference.llamacpp_install import pip_install_strategies

    strategies = pip_install_strategies(prefer_cuda=False)
    assert all("cu124" not in " ".join(cmd) for cmd in strategies)


def test_pip_install_strategies_includes_windows_cuda_indexes(monkeypatch):
    from seiso.inference.llamacpp_install import pip_install_strategies

    monkeypatch.setattr("seiso.inference.llamacpp_install.sys.platform", "win32")
    strategies = pip_install_strategies(prefer_cuda=True)
    assert "https://abetlen.github.io/llama-cpp-python/whl/cu124" in strategies[0]


def test_ensure_reinstalls_when_cpu_wheel_on_nvidia(monkeypatch):
    from seiso.inference import llamacpp_install

    state = {"gpu": False, "install": 0}

    monkeypatch.setattr(llamacpp_install, "nvidia_hardware_visible", lambda: True)
    monkeypatch.setattr(llamacpp_install, "llamacpp_import_ok", lambda: (True, None))
    monkeypatch.setattr(
        llamacpp_install,
        "llamacpp_gpu_offload_supported",
        lambda: state["gpu"],
    )

    def fake_install(**kwargs):
        state["install"] += 1
        state["gpu"] = True
        return True

    monkeypatch.setattr(llamacpp_install, "pip_install_llamacpp", fake_install)

    result = llamacpp_install.ensure_llamacpp_installed(auto_install=True)
    assert state["install"] == 1
    assert result["llamacpp"] is True
    assert result["gpu_offload"] is True


def test_ensure_skips_install_when_cuda_capable(monkeypatch):
    from seiso.inference import llamacpp_install

    monkeypatch.setattr(llamacpp_install, "nvidia_hardware_visible", lambda: True)
    monkeypatch.setattr(llamacpp_install, "llamacpp_import_ok", lambda: (True, None))
    monkeypatch.setattr(llamacpp_install, "llamacpp_gpu_offload_supported", lambda: True)

    def fail_install(**kwargs):
        raise AssertionError("should not install")

    monkeypatch.setattr(llamacpp_install, "pip_install_llamacpp", fail_install)

    result = llamacpp_install.ensure_llamacpp_installed(auto_install=True)
    assert result["llamacpp"] is True
    assert result["installed"] is False
    assert result["gpu_offload"] is True


def test_nvidia_llamacpp_stack_integration(monkeypatch):
    """nvidia-smi path wires through install, model_pool, profile, and platform caps."""
    import os
    import platform as plat

    from seiso.hardware.gpus import clear_gpu_enumeration_cache
    from seiso.hardware.profile import detect_gpus
    from seiso.inference.model_pool import llama_load_kwargs
    from seiso.kernels.platform import GpuVendor, detect_gpu
    from seiso.memory.platform_profile import apply_platform_memory_profile
    from seiso.training.platform_caps import training_capabilities

    fake_gpus = [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "vram_total_mb": 24564,
            "vram_used_mb": None,
            "utilization_pct": None,
            "temperature_c": None,
        }
    ]

    monkeypatch.setattr("seiso.inference.model_pool._cuda_available", lambda: False)
    monkeypatch.setattr("seiso.inference.model_pool._nvidia_hardware_visible", lambda: True)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    monkeypatch.delenv("SEISO_LLAMA_GPU_LAYERS", raising=False)
    assert llama_load_kwargs(4096)["n_gpu_layers"] == -1

    monkeypatch.setattr("seiso.hardware.gpus._torch_gpus", lambda: [])
    monkeypatch.setattr("seiso.hardware.gpus._nvidia_smi_gpus", lambda: fake_gpus)
    monkeypatch.setattr("seiso.hardware.gpus._mlx_apple_gpu", lambda: [])
    monkeypatch.setattr("seiso.hardware.profile._nvidia_smi_metrics", lambda: {})
    clear_gpu_enumeration_cache()
    detect_gpu.cache_clear()
    training_capabilities.cache_clear()

    assert len(detect_gpus()) == 1
    gpu = detect_gpu()
    caps = training_capabilities()
    assert gpu.vendor == GpuVendor.NVIDIA
    assert caps["nvidia_hardware"] is True
    assert caps["gpu_count"] == 1

    monkeypatch.setattr(plat, "system", lambda: "Linux")
    monkeypatch.setattr(
        "seiso.training.platform_caps.training_capabilities",
        lambda: {
            "nvidia_hardware": True,
            "gpu_count": 1,
            "vendor": "nvidia",
            "train_platform": "cpu",
            "supports_mlx_inference": False,
        },
    )
    monkeypatch.delenv("SEISO_LLAMA_GPU_LAYERS", raising=False)
    apply_platform_memory_profile(profile={"backend": "torch", "gpus": fake_gpus, "ram_gb": 32})
    assert os.environ.get("SEISO_LLAMA_GPU_LAYERS") == "-1"
