"""Inference performance profiles: safe | interactive | throughput.

Profiles only *seed* defaults via ``setdefault`` so explicit env overrides win.
Call :func:`apply_inference_profile` once at process startup (Forge / CLI).
"""

from __future__ import annotations

import os
from typing import Literal

from seiso.env import env_str

InferenceProfile = Literal["safe", "interactive", "throughput"]

_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "safe": {
        "SEISO_SIDECAR_PERF_MODE": "0",
        "SEISO_STREAM_BATCH_CHARS": "16",
        "SEISO_OLLAMA_KEEP_ALIVE": "1m",
        "SEISO_LLAMA_FLASH_ATTN": "false",
        "SEISO_LLAMASWAP_SEND_NUM_CTX": "0",
    },
    "interactive": {
        # Defaults already tuned for chat; keep light seeds only.
        "SEISO_STREAM_BATCH_CHARS": "4",
        "SEISO_SIDECAR_PERF_MODE": "0",
    },
    "throughput": {
        "SEISO_SIDECAR_PERF_MODE": "1",
        "SEISO_STREAM_BATCH_CHARS": "8",
        "SEISO_OLLAMA_KEEP_ALIVE": "30m",
        "SEISO_LLAMASWAP_SEND_NUM_CTX": "1",
        "SEISO_INFERENCE_FUSED_KERNELS": "1",
        "SEISO_TORCH_CACHE_IMPLEMENTATION": "static",
    },
}


def resolve_inference_profile(raw: str | None = None) -> InferenceProfile:
    text = (raw if raw is not None else env_str("SEISO_INFERENCE_PROFILE", "interactive")).strip().lower()
    if text in {"safe", "interactive", "throughput"}:
        return text  # type: ignore[return-value]
    return "interactive"


def apply_inference_profile(profile: str | None = None) -> InferenceProfile:
    """Seed env defaults for the selected profile. Explicit env always wins."""
    resolved = resolve_inference_profile(profile)
    for key, value in _PROFILE_DEFAULTS[resolved].items():
        os.environ.setdefault(key, value)
    return resolved


def profile_sidecar_keep_alive_override() -> str | None:
    """Optional keep_alive while a chat/preload request is active."""
    profile = resolve_inference_profile()
    if profile == "throughput":
        return "30m"
    if profile == "interactive":
        return "5m"
    return None  # safe: use adaptive short residency
