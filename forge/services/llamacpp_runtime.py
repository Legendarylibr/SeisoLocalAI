"""Ensure llama-cpp-python is importable for GGUF chat."""

from __future__ import annotations

import logging
from typing import Any

from seiso.inference import llamacpp_install

logger = logging.getLogger(__name__)

__all__ = ["ensure_llamacpp_runtime", "llamacpp_import_ok"]


def llamacpp_import_ok() -> tuple[bool, str | None]:
    return llamacpp_install.llamacpp_import_ok()


def llamacpp_gpu_offload_supported() -> bool:
    return llamacpp_install.llamacpp_gpu_offload_supported()


def nvidia_hardware_visible() -> bool:
    return llamacpp_install.nvidia_hardware_visible()


def ensure_llamacpp_installed(*, auto_install: bool | None = None) -> dict[str, Any]:
    return llamacpp_install.ensure_llamacpp_installed(auto_install=auto_install)


def ensure_llamacpp_runtime(*, auto_install: bool | None = None) -> dict[str, Any]:
    """Forge startup hook — delegates to shared install logic."""
    ok, error = llamacpp_import_ok()
    if ok:
        gpu_offload = llamacpp_gpu_offload_supported()
        if not nvidia_hardware_visible() or gpu_offload:
            return {
                "llamacpp": True,
                "installed": False,
                "error": None,
                "gpu_offload": gpu_offload,
            }

    result = ensure_llamacpp_installed(auto_install=auto_install)
    return {
        "llamacpp": result["llamacpp"],
        "installed": result["installed"],
        "error": result["error"],
        "gpu_offload": llamacpp_gpu_offload_supported() if result["llamacpp"] else False,
    }
