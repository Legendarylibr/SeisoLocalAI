"""Apple MLX model loader (macOS)."""

from __future__ import annotations

import logging
from typing import Any

from seiso.models.loader import LoadOptions

logger = logging.getLogger(__name__)


def load_mlx(options: LoadOptions) -> tuple[Any, Any]:
    try:
        from mlx_lm import load as mlx_load
    except ImportError as exc:
        raise ImportError(
            "mlx-lm is required for MLX backend. Install with: pip install mlx-lm"
        ) from exc

    logger.info("Loading MLX model: %s", options.model_id)
    model, tokenizer = mlx_load(options.model_id)
    return model, tokenizer
