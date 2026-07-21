"""Rollout backend name resolution and config validation."""

from __future__ import annotations

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.rollout_http import resolve_vllm_base_url

_ROLLOUT_BACKENDS = frozenset({"hf", "sglang", "vllm", "auto"})
_HTTP_ROLLOUT_BACKENDS = frozenset({"sglang", "vllm"})


def _normalize_backend_name(name: str) -> str:
    key = str(name or "hf").lower().strip()
    if key == "data_gen":
        raise ValueError(
            "rollout_backend=data_gen is no longer supported; use rollout_backend=hf "
            "for colocated Hugging Face generate"
        )
    return key


def resolve_rollout_backend(
    config: SingleGpuSlimeConfig,
    *,
    world_size: int = 1,
) -> str:
    """Resolve effective backend.

    * ``hf`` (default) — colocated Hugging Face generate
    * ``sglang`` — OpenAI-compatible SGLang HTTP
    * ``vllm`` — OpenAI-compatible vLLM HTTP (multi-GPU TP server)
    * ``auto`` — prefer vLLM then SGLang when a base URL is set and
      ``world_size > 1``; otherwise HF
    """
    name = _normalize_backend_name(getattr(config, "rollout_backend", "hf") or "hf")
    if name not in {"hf", "sglang", "vllm", "auto"}:
        raise ValueError(f"rollout_backend must be one of: hf, sglang, vllm, auto (got {name!r})")
    if name == "auto":
        if world_size > 1:
            if resolve_vllm_base_url(config):
                return "vllm"
            if str(getattr(config, "sglang_base_url", "") or "").strip():
                return "sglang"
        return "hf"
    return name


def validate_rollout_backend_config(config: SingleGpuSlimeConfig) -> None:
    name = _normalize_backend_name(getattr(config, "rollout_backend", "hf") or "hf")
    if name not in {"hf", "sglang", "vllm", "auto"}:
        raise ValueError(f"rollout_backend must be one of: hf, sglang, vllm, auto (got {name!r})")
    if name == "sglang":
        base = str(getattr(config, "sglang_base_url", "") or "").strip()
        if not base:
            raise ValueError(
                "rollout_backend=sglang requires sglang_base_url (e.g. http://127.0.0.1:30000)"
            )
    if name == "vllm":
        base = resolve_vllm_base_url(config)
        if not base:
            raise ValueError(
                "rollout_backend=vllm requires vllm_base_url "
                "(e.g. http://127.0.0.1:8000), or a running managed multi-GPU "
                "vLLM server (SEISO_MANAGED_VLLM_ENABLED=true)"
            )
    timeout = float(getattr(config, "sglang_timeout_s", 120.0) or 120.0)
    if timeout <= 0:
        raise ValueError("sglang_timeout_s must be positive")
    max_workers = int(getattr(config, "sglang_max_workers", 8) or 8)
    if max_workers < 1:
        raise ValueError("sglang_max_workers must be positive")
    vllm_timeout = float(getattr(config, "vllm_timeout_s", 120.0) or 120.0)
    if vllm_timeout <= 0:
        raise ValueError("vllm_timeout_s must be positive")
    vllm_workers = int(getattr(config, "vllm_max_workers", 8) or 8)
    if vllm_workers < 1:
        raise ValueError("vllm_max_workers must be positive")
    mode = str(getattr(config, "vllm_weight_mode", "auto") or "auto").lower()
    if mode not in {"auto", "lora", "full"}:
        raise ValueError("vllm_weight_mode must be one of: auto, lora, full")
    if int(getattr(config, "vllm_weight_keep", 2) or 2) < 1:
        raise ValueError("vllm_weight_keep must be >= 1")
