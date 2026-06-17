"""Model loading abstractions — dispatches to torch/MLX backends."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ModelKind(enum.StrEnum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDING = "embedding"


class Backend(enum.StrEnum):
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
    device_map: str = "auto"
    use_flash_attention: bool = True


def detect_backend() -> Backend:
    """Pick the best available backend for the current platform."""
    try:
        import torch

        if torch.cuda.is_available():
            return Backend.TORCH
    except ImportError:
        pass

    try:
        import mlx.core  # noqa: F401

        return Backend.MLX
    except ImportError:
        pass

    return Backend.CPU


def load_model(options: LoadOptions) -> tuple[Any, Any]:
    """
    Load model + tokenizer/processor.
    Returns (model, tokenizer) — actual types depend on backend.
    """
    backend = detect_backend()
    logger.info("Loading %s via %s", options.model_id, backend.value)

    if backend == Backend.MLX:
        from seiso.models.mlx_loader import load_mlx

        return load_mlx(options)

    from seiso.models.torch_loader import load_torch

    return load_torch(options, backend=backend)
