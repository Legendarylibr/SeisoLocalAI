"""Model loading abstractions — dispatches to torch/MLX backends."""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from seiso.compat import StrEnum

logger = logging.getLogger(__name__)


class ModelKind(StrEnum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDING = "embedding"


class Backend(StrEnum):
    TORCH = "torch"
    MLX = "mlx"
    CPU = "cpu"


@dataclass(frozen=True)
class LoadOptions:
    model_id: str
    kind: ModelKind = ModelKind.TEXT
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_seq_length: int = 4096
    dtype: str | None = None
    trust_remote_code: bool = False
    device_map: str | dict[str, str] | None = "auto"
    use_flash_attention: bool = True
    revision: str | None = None


def load_mlx(options: LoadOptions) -> tuple[Any, Any]:
    """Load a model via MLX (lazy import to avoid unconditional dependency)."""
    from seiso.models.mlx_loader import load_mlx as _load_mlx

    return _load_mlx(options)


@lru_cache(maxsize=1)
def detect_backend() -> Backend:
    """Pick the best available backend for inference."""
    try:
        import torch

        if torch.cuda.is_available():
            return Backend.TORCH
    except ImportError:
        pass

    if platform.system().lower() in {"linux", "windows"}:
        try:
            from seiso.security.nvidia_boundary import nvidia_smi_visible

            if nvidia_smi_visible():
                return Backend.TORCH
        except ImportError:
            pass

    if os.environ.get("SEISO_SKIP_MLX_PROBE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        try:
            import mlx.core  # noqa: F401

            return Backend.MLX
        except ImportError:
            pass

    return Backend.CPU


def detect_training_device() -> str:
    """Resolve PyTorch device string for training (never MLX)."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(options: LoadOptions, *, for_training: bool = False) -> tuple[Any, Any]:
    """
    Load model + tokenizer/processor.
    Returns (model, tokenizer) — actual types depend on backend.

    Training always uses PyTorch (CUDA, ROCm, MPS, or CPU) — never MLX.
    """
    if for_training:
        from seiso.models.torch_loader import load_torch

        device = detect_training_device()
        backend = Backend.TORCH if device in ("cuda", "mps") else Backend.CPU
        logger.info("Loading %s for training via torch (%s)", options.model_id, device)
        return load_torch(options, backend=backend, device=device, for_training=True)

    backend = detect_backend()
    logger.info("Loading %s via %s", options.model_id, backend.value)

    if backend == Backend.MLX:
        return load_mlx(options)

    from seiso.models.torch_loader import load_torch

    return load_torch(options, backend=backend)
