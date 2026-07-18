"""Flash Attention 2/3 and SDPA resolution for training and inference."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def _env_prefer_fa3() -> bool:
    raw = os.environ.get("SEISO_PREFER_FLASH_ATTN3", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _env_force_attn() -> str | None:
    """Optional force: SEISO_ATTN_IMPLEMENTATION=sdpa|flash_attention_2|eager."""
    raw = os.environ.get("SEISO_ATTN_IMPLEMENTATION", "").strip().lower()
    if not raw:
        return None
    aliases = {
        "fa2": "flash_attention_2",
        "flash": "flash_attention_2",
        "flash_attn": "flash_attention_2",
        "flash_attention_2": "flash_attention_2",
        "fa3": "flash_attention_3",
        "flash_attention_3": "flash_attention_3",
        "sdpa": "sdpa",
        "eager": "eager",
        "math": "eager",
    }
    return aliases.get(raw, raw)


@lru_cache(maxsize=1)
def resolve_attention_implementation(*, prefer_fa3: bool | None = None) -> str:
    """
    Pick the best attention backend available on this GPU.

    Priority (Hopper+): flash_attention_3 → flash_attention_2 → sdpa → eager
    Priority (Ampere/Ada): flash_attention_2 → sdpa → eager

    Override with ``SEISO_ATTN_IMPLEMENTATION`` or disable FA3 prefer via
    ``SEISO_PREFER_FLASH_ATTN3=0``.
    """
    forced = _env_force_attn()
    if forced:
        logger.info("Using attention implementation from env: %s", forced)
        return forced

    if prefer_fa3 is None:
        prefer_fa3 = _env_prefer_fa3()

    try:
        import torch

        if not torch.cuda.is_available():
            return "sdpa"
        major, _minor = torch.cuda.get_device_capability(0)
        is_hopper_plus = major >= 9
    except ImportError:
        return "sdpa"

    if prefer_fa3 and is_hopper_plus:
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


def enable_torch_sdpa_backends(*, deterministic: bool = False) -> dict[str, bool]:
    """Prefer flash/mem-efficient SDPA kernels when available (CUDA).

    Safe no-op on CPU/MPS or older torch. Does not install flash-attn.
    """
    flags = {
        "flash": False,
        "mem_efficient": False,
        "math": True,
        "cudnn": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return flags
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return flags
        # Enable backends; torch picks the best per shape.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(not deterministic)
            flags["flash"] = not deterministic
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            flags["mem_efficient"] = True
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
            flags["math"] = True
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            try:
                torch.backends.cuda.enable_cudnn_sdp(not deterministic)
                flags["cudnn"] = not deterministic
            except Exception:
                pass
    except ImportError:
        pass
    return flags


def attention_metadata() -> dict[str, Any]:
    impl = resolve_attention_implementation()
    sdpa_flags = enable_torch_sdpa_backends()
    return {
        "attn_implementation": impl,
        "flash_attention_3": impl == "flash_attention_3",
        "flash_attention_2": impl == "flash_attention_2",
        "sdpa": impl == "sdpa",
        "eager": impl == "eager",
        "sdpa_backends": sdpa_flags,
        "recommended": impl if impl != "eager" else "sdpa",
    }


def attention_doctor_lines() -> list[str]:
    """Human-readable lines for doctor / CLI diagnostics."""
    meta = attention_metadata()
    impl = str(meta["attn_implementation"])
    lines = [f"attention: {impl}"]
    if meta.get("flash_attention_2") or meta.get("flash_attention_3"):
        lines.append("  flash-attn package path active (best for long context)")
    elif meta.get("sdpa"):
        lines.append(
            "  SDPA (PyTorch) — install flash-attn for more long-context speed: "
            "./scripts/install_flash_attn.sh"
        )
    else:
        lines.append("  eager attention — enable CUDA + SDPA for training speed")
    backends = meta.get("sdpa_backends") or {}
    if isinstance(backends, dict) and backends:
        on = [k for k, v in backends.items() if v]
        if on:
            lines.append(f"  torch SDPA backends enabled: {', '.join(on)}")
    return lines
