"""VRAM and download-size heuristics — core library, no Forge dependency."""

from __future__ import annotations

import math
import re

from seiso.models.catalog import _parse_param_size

_UNKNOWN_PARAMS_B = 7.0

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b", re.I)
_ACTIVE_MOE_RE = re.compile(r"a(\d+(?:\.\d+)?)b", re.I)


def guess_params_from_name(name: str) -> float | None:
    """Extract parameter count (billions) from a model name or path."""
    m = _PARAM_RE.search(name)
    return float(m.group(1)) if m else None


def _active_params_b(
    params: str, tags: tuple[str, ...] | list[str], repo_id: str = ""
) -> float:
    """Effective parameter count for VRAM estimates (MoE / active experts)."""
    text = f"{params} {repo_id}".lower()
    moe_match = _ACTIVE_MOE_RE.search(text)
    if moe_match:
        return float(moe_match.group(1))
    try:
        raw = _parse_param_size(params)
    except ValueError:
        raw = float("nan")
    if not math.isfinite(raw):
        guessed = guess_params_from_name(repo_id) or guess_params_from_name(params)
        raw = guessed if guessed is not None else _UNKNOWN_PARAMS_B
    if "moe" in tags or "moe" in text:
        return max(raw * 0.2, 1.0)
    return raw


def _quant_bytes_per_param_b(quant: str) -> float:
    quant_u = quant.upper()
    if "Q8" in quant_u or "F16" in quant_u or "BF16" in quant_u:
        return 1.1
    if "Q5" in quant_u:
        return 0.75
    if "Q4" in quant_u or "IQ4" in quant_u:
        return 0.55
    return 0.65


def estimate_chat_vram_gb(
    params: str,
    *,
    quant: str = "Q4_K_M",
    tags: tuple[str, ...] | list[str] = (),
    repo_id: str = "",
) -> float:
    """Rough GGUF chat VRAM — conservative, for fit labels only."""
    params_b = _active_params_b(params, tags, repo_id)
    return round(params_b * _quant_bytes_per_param_b(quant) + 1.2, 2)


def estimate_safetensors_download_bytes(
    params: str,
    *,
    tags: tuple[str, ...] | list[str] = (),
    repo_id: str = "",
) -> int:
    """Estimate on-disk safetensors size from parameter count (bf16-ish)."""
    params_b = _active_params_b(params, tags, repo_id)
    gb = params_b * 2.0 + 0.5
    gb = min(max(gb, 0.2), 10_000.0)
    return int(gb * 1024**3)


def estimate_training_vram_gb(
    params: str,
    *,
    quant: str = "4bit",
    tags: tuple[str, ...] | list[str] = (),
    repo_id: str = "",
) -> float:
    """Rough QLoRA/LoRA training VRAM for fit labels."""
    params_b = _active_params_b(params, tags, repo_id)
    quant_u = quant.lower()
    if quant_u in {"mxfp4", "fp8"}:
        # Native hub quant (~8-bit effective weights).
        base = params_b * 1.1
    elif quant_u in {"16bit", "none", "fp16", "bf16"}:
        base = params_b * 2.0
    elif quant_u == "8bit":
        base = params_b * 1.1
    else:
        base = params_b * 0.55
    return round(base * 2.2 + 1.5, 2)


def estimate_gguf_download_bytes(
    params: str,
    *,
    quant: str = "Q4_K_M",
    tags: tuple[str, ...] | list[str] = (),
    repo_id: str = "",
) -> int:
    """Estimate on-disk GGUF size from active params and quant."""
    params_b = _active_params_b(params, tags, repo_id)
    gb = params_b * _quant_bytes_per_param_b(quant) + 0.4
    gb = min(max(gb, 0.25), 10_000.0)
    return int(gb * 1024**3)
