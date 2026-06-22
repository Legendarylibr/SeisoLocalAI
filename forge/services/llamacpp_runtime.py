"""Ensure llama-cpp-python is importable for GGUF chat."""

from __future__ import annotations

import logging
from typing import Any

from seiso.inference.llamacpp_install import (
    ensure_llamacpp_installed,
    llamacpp_import_ok,
)

logger = logging.getLogger(__name__)

__all__ = ["ensure_llamacpp_runtime", "llamacpp_import_ok"]


def ensure_llamacpp_runtime(*, auto_install: bool | None = None) -> dict[str, Any]:
    """Forge startup hook — delegates to shared install logic."""
    result = ensure_llamacpp_installed(auto_install=auto_install)
    return {
        "llamacpp": result["llamacpp"],
        "installed": result["installed"],
        "error": result["error"],
    }
