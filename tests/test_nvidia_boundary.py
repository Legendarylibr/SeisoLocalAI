"""NVIDIA secure boundary and WSL2 platform tests."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from seiso.kernels.platform import detect_gpu
from seiso.security.nvidia_boundary import (
    detect_wsl2,
    enforce_nvidia_secure_boundary,
    nvidia_boundary_report,
    nvidia_smi_visible,
    query_nvidia_gpus,
)


def test_detect_wsl2_env():
    with patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=False):
        assert detect_wsl2() is True


def test_nvidia_boundary_report_keys():
    report = nvidia_boundary_report()
    assert "linux_nvidia_host" in report
    assert "wsl2" in report
    assert "approved_tier" in report


def test_enforce_boundary_skips_non_linux():
    with patch(
        "seiso.security.nvidia_boundary.is_linux_nvidia_host", return_value=False
    ):
        report = enforce_nvidia_secure_boundary(context="test")
    assert report["linux_nvidia_host"] is False


def test_enforce_boundary_requires_ack_on_linux_nvidia():
    env_clear = {
        "SEISO_NVIDIA_HOST_VENV_ACK": "",
        "SEISO_NVIDIA_SECURE_VM": "",
        "SEISO_NVIDIA_WSL_ACK": "",
        "ADAPTIVE_RL_NVIDIA_HOST_VENV_ACK": "",
        "ADAPTIVE_RL_NVIDIA_SECURE_VM": "",
        "ADAPTIVE_RL_NVIDIA_WSL_ACK": "",
    }
    with patch(
        "seiso.security.nvidia_boundary.is_linux_nvidia_host", return_value=True
    ):
        with patch("seiso.security.nvidia_boundary.in_ci", return_value=False):
            with patch.dict(os.environ, env_clear, clear=False):
                with pytest.raises(SystemExit):
                    enforce_nvidia_secure_boundary(context="test")


def test_enforce_boundary_wsl2_ack():
    env = {
        "SEISO_NVIDIA_WSL_ACK": "1",
        "WSL_DISTRO_NAME": "Ubuntu",
    }
    with patch(
        "seiso.security.nvidia_boundary.is_linux_nvidia_host", return_value=True
    ):
        with patch("seiso.security.nvidia_boundary.in_ci", return_value=False):
            with patch.dict(os.environ, env, clear=False):
                report = enforce_nvidia_secure_boundary(context="test")
    assert report.get("approved_tier") == "wsl2"


def test_forge_startup_sets_native_linux_boundary_for_local_training(monkeypatch):
    from forge.security.startup import (
        _LEGACY_NVIDIA_HOST_VENV_ACK_ENV,
        _NVIDIA_HOST_VENV_ACK_ENV,
        _set_native_linux_nvidia_boundary_for_local_forge,
    )

    monkeypatch.delenv(_NVIDIA_HOST_VENV_ACK_ENV, raising=False)
    monkeypatch.delenv(_LEGACY_NVIDIA_HOST_VENV_ACK_ENV, raising=False)
    monkeypatch.setattr("seiso.security.nvidia_boundary.detect_wsl2", lambda: False)
    monkeypatch.setattr("seiso.security.nvidia_boundary.in_ci", lambda: False)
    monkeypatch.setattr(
        "seiso.security.nvidia_boundary.is_linux_nvidia_host", lambda: True
    )
    monkeypatch.setattr(
        "seiso.security.nvidia_boundary.approved_nvidia_boundary", lambda: None
    )

    settings = SimpleNamespace(allow_remote=False)

    assert _set_native_linux_nvidia_boundary_for_local_forge(settings) is True
    assert os.environ[_NVIDIA_HOST_VENV_ACK_ENV] == "1"
    assert os.environ[_LEGACY_NVIDIA_HOST_VENV_ACK_ENV] == "1"


def test_forge_startup_does_not_auto_ack_remote_nvidia(monkeypatch):
    from forge.security.startup import (
        _NVIDIA_HOST_VENV_ACK_ENV,
        _set_native_linux_nvidia_boundary_for_local_forge,
    )

    monkeypatch.delenv(_NVIDIA_HOST_VENV_ACK_ENV, raising=False)
    monkeypatch.setattr("seiso.security.nvidia_boundary.detect_wsl2", lambda: False)
    monkeypatch.setattr("seiso.security.nvidia_boundary.in_ci", lambda: False)
    monkeypatch.setattr(
        "seiso.security.nvidia_boundary.is_linux_nvidia_host", lambda: True
    )
    monkeypatch.setattr(
        "seiso.security.nvidia_boundary.approved_nvidia_boundary", lambda: None
    )

    settings = SimpleNamespace(allow_remote=True)

    assert _set_native_linux_nvidia_boundary_for_local_forge(settings) is False
    assert _NVIDIA_HOST_VENV_ACK_ENV not in os.environ


def test_gpu_platform_wsl2_field():
    platform = detect_gpu()
    assert hasattr(platform, "is_wsl2")
    assert isinstance(platform.is_wsl2, bool)


def test_query_nvidia_gpus_csv_fallback(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_run(exe: str, *args: str, timeout: float = 10.0):
        calls.append(args)
        if args[:2] == (
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ):
            return type(
                "Proc",
                (),
                {"returncode": 1, "stdout": "", "stderr": ""},
            )()
        if args[:2] == (
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ):
            return type(
                "Proc",
                (),
                {
                    "returncode": 0,
                    "stdout": "NVIDIA GeForce RTX 4090, 24564\n",
                    "stderr": "",
                },
            )()
        return type("Proc", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(
        "seiso.security.nvidia_boundary.resolve_nvidia_smi_executable",
        lambda: "/usr/bin/nvidia-smi",
    )
    monkeypatch.setattr("seiso.security.nvidia_boundary._run_nvidia_smi", fake_run)

    gpus = query_nvidia_gpus()
    assert len(gpus) == 1
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 4090"
    assert gpus[0]["memory_total_mb"] == 24564
    assert nvidia_smi_visible() is True


def test_detect_gpu_nvidia_smi_fallback(monkeypatch):
    from seiso.hardware.gpus import clear_gpu_enumeration_cache

    detect_gpu.cache_clear()
    clear_gpu_enumeration_cache()
    monkeypatch.setattr(
        "seiso.security.nvidia_boundary.query_nvidia_gpus",
        lambda **kwargs: [
            {"index": 0, "name": "NVIDIA GeForce RTX 4090", "memory_total_mb": 24564}
        ],
    )
    monkeypatch.setattr("seiso.hardware.probes.nvidia.platform.system", lambda: "Linux")
    monkeypatch.setattr("seiso.hardware.gpus._torch_gpus", lambda: [])
    clear_gpu_enumeration_cache()

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()
        version = type("Version", (), {"hip": None})()

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch())
    detect_gpu.cache_clear()

    platform = detect_gpu()
    assert platform.vendor.value == "nvidia"
    assert platform.device_count == 1
    assert "4090" in platform.device_name
    detect_gpu.cache_clear()
    clear_gpu_enumeration_cache()


def test_platform_caps_install_hint_without_cuda_runtime(monkeypatch):
    from seiso.kernels.platform import GpuPlatform, GpuVendor, detect_gpu
    from seiso.training.platform_caps import training_capabilities

    training_capabilities.cache_clear()
    detect_gpu.cache_clear()
    monkeypatch.setattr(
        "seiso.training.platform_caps.detect_gpu",
        lambda: GpuPlatform(
            vendor=GpuVendor.NVIDIA,
            device_name="NVIDIA GeForce RTX 4090",
            device_count=1,
            supports_native_cuda=False,
            supports_triton=False,
        ),
    )

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()
        library = type("Library", (), {})()
        backends = type(
            "Backends",
            (),
            {"mps": type("Mps", (), {"is_available": staticmethod(lambda: False)})()},
        )()

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch())
    monkeypatch.setitem(
        __import__("sys").modules,
        "bitsandbytes",
        __import__("types").ModuleType("bitsandbytes"),
    )
    monkeypatch.setattr("seiso.training.platform_caps.platform.system", lambda: "Linux")

    caps = training_capabilities()
    assert caps["gpu_count"] == 1
    assert caps["nvidia_hardware"] is True
    assert caps["cuda_runtime"] is False
    assert caps["train_platform"] == "cpu"
    assert "cuda" in caps["install_extra"]
    training_capabilities.cache_clear()
    detect_gpu.cache_clear()
