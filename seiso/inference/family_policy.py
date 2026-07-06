"""GGUF model-family policy for llama.cpp inference safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FamilyKind = Literal["dense", "swa", "moe"]


@dataclass(frozen=True, slots=True)
class InferenceFamilyPolicy:
    architecture: str
    kind: FamilyKind
    allow_partial_offload: bool
    allow_flash_attn: bool
    allow_kv_quant: bool
    swa_full_default: bool
    prefill_tightness: float


_DENSE_ARCH_MARKERS = (
    "llama",
    "qwen2",
    "qwen3",
    "qwen",
    "mistral",
    "gemma2",
    "phi",
    "phi3",
    "phi-3",
    "olmo",
    "granite",
    "internlm",
    "baichuan",
    "yi",
)


def _dense_prefill_tightness(architecture: str) -> float:
    arch = architecture.lower()
    if any(marker in arch for marker in ("qwen3", "qwen2", "qwen")):
        return 1.20
    return 1.0


def policy_for_gguf(model_path: str) -> InferenceFamilyPolicy:
    """Return family policy from GGUF metadata, falling back conservatively."""
    try:
        from seiso.inference import backends

        architecture = (backends.gguf_architecture(model_path) or "").lower()
        uses_swa = backends.gguf_uses_sliding_window_attention(model_path)
        is_moe = backends.gguf_is_moe(model_path)
    except Exception:
        architecture = ""
        uses_swa = False
        is_moe = False

    if uses_swa:
        return InferenceFamilyPolicy(
            architecture=architecture,
            kind="swa",
            allow_partial_offload=False,
            allow_flash_attn=False,
            allow_kv_quant=False,
            swa_full_default=False,
            prefill_tightness=1.25,
        )
    if is_moe:
        return InferenceFamilyPolicy(
            architecture=architecture,
            kind="moe",
            allow_partial_offload=True,
            allow_flash_attn=False,
            allow_kv_quant=False,
            swa_full_default=True,
            prefill_tightness=1.15,
        )

    known_dense = any(marker in architecture for marker in _DENSE_ARCH_MARKERS)
    return InferenceFamilyPolicy(
        architecture=architecture,
        kind="dense",
        allow_partial_offload=True,
        allow_flash_attn=True,
        allow_kv_quant=True,
        swa_full_default=True,
        prefill_tightness=_dense_prefill_tightness(architecture) if known_dense else 1.0,
    )
