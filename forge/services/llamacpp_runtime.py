"""Ensure llama-cpp-python is importable for GGUF chat."""

from __future__ import annotations

import logging
from typing import Any

from seiso.inference.llamacpp_install import (
    ensure_llamacpp_installed,
    llamacpp_gpu_offload_supported,
    llamacpp_import_ok,
    nvidia_hardware_visible,
)

logger = logging.getLogger(__name__)

__all__ = ["ensure_llamacpp_runtime", "llamacpp_import_ok"]


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
            }

    result = ensure_llamacpp_installed(auto_install=auto_install)
    return {
        "llamacpp": result["llamacpp"],
        "installed": result["installed"],
        "error": result["error"],
    }
