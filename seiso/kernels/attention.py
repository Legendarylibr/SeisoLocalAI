"""Flash Attention 2/3 resolution for training and inference."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def resolve_attention_implementation(*, prefer_fa3: bool = True) -> str:
    """
    Pick the best attention backend available on this GPU.

    Priority (Hopper+): flash_attention_3 → flash_attention_2 → sdpa → eager
    Priority (Ampere/Ada): flash_attention_2 → sdpa → eager
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "sdpa"
        major, _minor = torch.cuda.get_device_capability(0)
        is_hopper_plus = major >= 9
    except ImportError:
        return "sdpa"

    if prefer_fa3 and is_hopper_plus:
        # FA3 via HuggingFace kernels hub (vLLM / varunneal builds)
        try:
            from kernels import get_kernel  # type: ignore[import-untyped]

            get_kernel("kernels-community/flash-attn3")
            logger.info("Using Flash Attention 3 (kernels hub)")
            return "flash_attention_3"
        except Exception:
            pass
        try:
            import flash_attn_interface  # type: ignore[import-untyped]  # noqa: F401

            logger.info("Using Flash Attention 3 (flash_attn_interface)")
            return "flash_attention_3"
        except ImportError:
            pass

    try:
        import flash_attn  # noqa: F401

        logger.info("Using Flash Attention 2")
        return "flash_attention_2"
    except ImportError:
        pass

    logger.info("Using SDPA attention (flash-attn not installed)")
    return "sdpa"


def attention_metadata() -> dict[str, str | bool]:
    impl = resolve_attention_implementation()
    return {
        "attn_implementation": impl,
        "flash_attention_3": impl == "flash_attention_3",
        "flash_attention_2": impl == "flash_attention_2",
        "sdpa": impl == "sdpa",
    }