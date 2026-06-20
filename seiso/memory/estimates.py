"""VRAM and download-size heuristics — core library, no Forge dependency."""

from __future__ import annotations

import re

from seiso.models.catalog import _parse_param_size

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b", re.I)
_ACTIVE_MOE_RE = re.compile(r"a(\d+(?:\.\d+)?)b", re.I)


def guess_params_from_name(name: str) -> float | None:
    """Extract parameter count (billions) from a model name or path."""
    m = _PARAM_RE.search(name)
    return float(m.group(1)) if m else None


def _active_params_b(params: str, tags: tuple[str, ...] | list[str], repo_id: str = "") -> float:
    """Effective parameter count for VRAM estimates (MoE / active experts)."""
    text = f"{params} {repo_id}".lower()
    moe_match = _ACTIVE_MOE_RE.search(text)
    if moe_match:
        return float(moe_match.group(1))
    raw = _parse_param_size(params)
    if "moe" in tags:
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
    return int(max(gb, 0.25) * 1024**3)
