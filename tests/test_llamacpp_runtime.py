"""Tests for llama-cpp-python runtime repair."""

from __future__ import annotations

from unittest.mock import patch


def test_llamacpp_import_ok_when_module_present():
    from forge.services.llamacpp_runtime import llamacpp_import_ok

    with patch.dict("sys.modules", {"llama_cpp": object()}):
        ok, error = llamacpp_import_ok()
    assert ok is True
    assert error is None


def test_ensure_llamacpp_runtime_skips_install_when_present():
    from forge.services.llamacpp_runtime import ensure_llamacpp_runtime

    with (
        patch(
            "forge.services.llamacpp_runtime.llamacpp_import_ok",
            return_value=(True, None),
        ),
        patch(
            "forge.services.llamacpp_runtime.nvidia_hardware_visible",
            return_value=False,
        ),
        patch(
            "forge.services.llamacpp_runtime.llamacpp_gpu_offload_supported",
            return_value=False,
        ),
        patch(
            "forge.services.llamacpp_runtime.ensure_llamacpp_installed",
            side_effect=AssertionError("should not install"),
        ),
    ):
        result = ensure_llamacpp_runtime(auto_install=True)
    assert result["llamacpp"] is True
    assert result["installed"] is False


def test_ensure_llamacpp_runtime_delegates_for_cpu_wheel_on_nvidia():
    from forge.services.llamacpp_runtime import ensure_llamacpp_runtime

    with (
        patch(
            "forge.services.llamacpp_runtime.llamacpp_import_ok",
            return_value=(True, None),
        ),
        patch(
            "forge.services.llamacpp_runtime.nvidia_hardware_visible", return_value=True
        ),
        patch(
            "forge.services.llamacpp_runtime.llamacpp_gpu_offload_supported",
            return_value=False,
        ),
        patch(
            "forge.services.llamacpp_runtime.ensure_llamacpp_installed",
            return_value={
                "llamacpp": True,
                "installed": True,
                "gpu_offload": True,
                "error": None,
            },
        ) as ensure_installed,
    ):
        result = ensure_llamacpp_runtime(auto_install=True)

    ensure_installed.assert_called_once_with(auto_install=True)
    assert result["llamacpp"] is True
    assert result["installed"] is True


def test_ensure_llamacpp_runtime_installs_when_missing(monkeypatch):
    from forge.services import hf_connectivity
    from forge.services.llamacpp_runtime import ensure_llamacpp_runtime

    hf_connectivity.check_inference_runtime.cache_clear()
    calls = {"install": 0}

    def fake_import():
        calls["install"] += 1
        return calls["install"] > 1, (
            None if calls["install"] > 1 else "ModuleNotFoundError: x"
        )

    monkeypatch.setattr(
        "forge.services.llamacpp_runtime.llamacpp_import_ok", fake_import
    )
    monkeypatch.setattr(
        "forge.services.llamacpp_runtime.ensure_llamacpp_installed",
        lambda **kwargs: {"llamacpp": True, "installed": True, "error": None},
    )

    result = ensure_llamacpp_runtime(auto_install=True)
    assert result["llamacpp"] is True
    assert result["installed"] is True
    hf_connectivity.check_inference_runtime.cache_clear()


def test_check_inference_runtime_reports_import_error(monkeypatch):
    from forge.services import hf_connectivity

    hf_connectivity.check_inference_runtime.cache_clear()
    monkeypatch.setattr(
        "forge.services.hf_connectivity._llamacpp_status",
        lambda: (False, "OSError: libggml.so missing"),
    )

    status = hf_connectivity.check_inference_runtime()
    assert status.llamacpp is False
    assert "libggml" in (status.llamacpp_error or "")
    assert any("Import error" in hint for hint in status.install_hints)
    hf_connectivity.check_inference_runtime.cache_clear()
