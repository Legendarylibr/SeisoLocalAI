"""GGUF model-family policy for llama.cpp inference safety."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FamilyKind = Literal["dense", "swa", "moe"]

# Baseline fp16 KV bytes/token for a 7B-class model — used to scale prefill risk.
_KV_TIGHTNESS_BASELINE_BYTES = 64 * 1024
_KV_TIGHTNESS_MAX = 1.35
_GEMMA_SWA_HINT_RE = re.compile(r"(^|[^a-z0-9])gemma[-_ ]?(3|4|3n)([^a-z0-9]|$)")
_MOE_HINT_RE = re.compile(
    r"(^|[^a-z0-9])(mixtral|qwen[-_ ]?\d+(?:\.\d+)?[-_ ]?moe|deepseek[-_ ]?(v2|2)|moe)([^a-z0-9]|$)"
)


@dataclass(frozen=True, slots=True)
class InferenceFamilyPolicy:
    kind: FamilyKind
    allow_partial_offload: bool
    allow_flash_attn: bool
    allow_kv_quant: bool
    swa_full_default: bool
    prefill_tightness: float


def _prefill_tightness_for_dense(model_path: str) -> float:
    """Scale prefill conservatism from measured KV geometry, not architecture names."""
    try:
        from seiso.inference.backends import gguf_kv_bytes_per_token

        kv_bytes = gguf_kv_bytes_per_token(model_path)
    except Exception:
        kv_bytes = None
    if not kv_bytes or kv_bytes <= 0:
        return 1.0
    excess = max(0, kv_bytes - _KV_TIGHTNESS_BASELINE_BYTES)
    span = max(_KV_TIGHTNESS_BASELINE_BYTES * 4, 1)
    return min(_KV_TIGHTNESS_MAX, 1.0 + excess / span)


def _model_hint_text(model_path: str, architecture: str) -> str:
    name = Path(model_path).name.lower()
    return f"{architecture.lower()} {name}"


def _looks_like_swa_family(model_path: str, architecture: str) -> bool:
    return bool(_GEMMA_SWA_HINT_RE.search(_model_hint_text(model_path, architecture)))


def _looks_like_moe_family(model_path: str, architecture: str) -> bool:
    return bool(_MOE_HINT_RE.search(_model_hint_text(model_path, architecture)))


def policy_for_gguf(model_path: str) -> InferenceFamilyPolicy:
    """Return family policy from GGUF metadata, falling back conservatively."""
    try:
        from seiso.inference import backends

        architecture = (backends.gguf_architecture(model_path) or "").lower()
        uses_swa = backends.gguf_uses_sliding_window_attention(
            model_path
        ) or _looks_like_swa_family(model_path, architecture)
        is_moe = backends.gguf_is_moe(model_path) or _looks_like_moe_family(
            model_path, architecture
        )
    except Exception:
        architecture = ""
        uses_swa = _looks_like_swa_family(model_path, architecture)
        is_moe = _looks_like_moe_family(model_path, architecture)

    if uses_swa:
        return InferenceFamilyPolicy(
            kind="swa",
            allow_partial_offload=False,
            allow_flash_attn=False,
            allow_kv_quant=False,
            swa_full_default=False,
            prefill_tightness=1.25,
        )
    if is_moe:
        tightness = max(1.1, _prefill_tightness_for_dense(model_path))
        tightness = min(_KV_TIGHTNESS_MAX, tightness)
        return InferenceFamilyPolicy(
            kind="moe",
            allow_partial_offload=True,
            allow_flash_attn=False,
            allow_kv_quant=False,
            swa_full_default=True,
            prefill_tightness=tightness,
        )

    return InferenceFamilyPolicy(
        kind="dense",
        allow_partial_offload=True,
        allow_flash_attn=True,
        allow_kv_quant=True,
        swa_full_default=True,
        prefill_tightness=_prefill_tightness_for_dense(model_path),
    )
