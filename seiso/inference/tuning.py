"""Process-wide and per-request tuning for fast local inference."""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any

from seiso.env import env_bool, env_int, env_str

logger = logging.getLogger(__name__)

_torch_configured = False
_torch_lock = threading.Lock()
_torch_compiled_ids: set[int] = set()


def configure_torch_inference() -> None:
    """Apply one-time PyTorch settings for faster inference (idempotent)."""
    global _torch_configured
    with _torch_lock:
        if _torch_configured:
            return
        try:
            import torch
        except ImportError:
            return

        if torch.cuda.is_available():
            if env_bool("SEISO_TORCH_CUDNN_BENCHMARK", True):
                torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(False)
            with contextlib.suppress(Exception):
                torch.set_float32_matmul_precision("high")

        _torch_configured = True
        logger.debug("PyTorch inference backends configured")


def prepare_torch_model(model: Any) -> Any:
    """Eval mode + KV cache + optional torch.compile for generation."""
    configure_torch_inference()
    model.eval()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = True
    return maybe_compile_torch_model(model)


def maybe_compile_torch_model(model: Any) -> Any:
    """Optionally compile the model graph for lower per-token latency on CUDA."""
    if not env_bool("SEISO_TORCH_COMPILE", False):
        return model
    model_id = id(model)
    if model_id in _torch_compiled_ids:
        return model
    try:
        import torch

        if not torch.cuda.is_available() or not hasattr(torch, "compile"):
            return model
        if getattr(model, "is_quantized", False) or hasattr(model, "peft_config"):
            return model
        mode = env_str("SEISO_TORCH_COMPILE_MODE", "reduce-overhead")
        compiled = torch.compile(model, mode=mode)
        _torch_compiled_ids.add(model_id)
        logger.info("torch.compile enabled for inference (mode=%s)", mode)
        return compiled
    except Exception as exc:
        logger.debug("torch.compile skipped: %s", exc)
        return model


def apply_inference_kernels(model: Any) -> None:
    """Patch CUDA weights with fused RMSNorm/SwiGLU when enabled."""
    if not env_bool("SEISO_INFERENCE_FUSED_KERNELS", True):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
    except ImportError:
        return
    try:
        from seiso.kernels.hooks import apply_training_kernels

        apply_training_kernels(model, use_cuda=True, use_triton=True, patch_mlp=True)
    except Exception as exc:
        logger.debug("Inference fused kernels skipped: %s", exc)


def build_mlx_sampler(payload: dict[str, Any]) -> Any | None:
    temperature = float(payload.get("temperature", 0.0))
    if temperature <= 0:
        return None
    try:
        from mlx_lm.sample_utils import make_sampler

        top_p = float(payload.get("top_p") or 0.0)
        return make_sampler(temp=temperature, top_p=top_p)
    except ImportError:
        return None


def _default_mlx_prefill_step() -> int:
    from seiso.memory.protection import headroom_mb

    headroom = headroom_mb()
    if headroom < 4096:
        return 512
    if headroom < 8192:
        return 1024
    if headroom < 16384:
        return 2048
    return 4096


def mlx_stream_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_tokens": payload.get("max_tokens", 512)}
    sampler = build_mlx_sampler(payload)
    if sampler is not None:
        kwargs["sampler"] = sampler
    prefill = env_int("SEISO_MLX_PREFILL_STEP", _default_mlx_prefill_step())
    if prefill > 0:
        kwargs["prefill_step_size"] = prefill
    kv_bits = env_int("SEISO_MLX_KV_BITS", 0)
    if kv_bits > 0:
        kwargs["kv_bits"] = kv_bits
        kwargs["kv_group_size"] = env_int("SEISO_MLX_KV_GROUP_SIZE", 64)
    return kwargs


def estimate_llama_n_ctx(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    default: int = 4096,
    floor: int = 2048,
    ceiling: int | None = None,
    model_path: str | None = None,
    model_format: str | None = None,
    model_name: str | None = None,
) -> int:
    """Right-size context window to prompt + generation (faster KV cache)."""
    from seiso.inference.context_limits import effective_context_ceiling

    if ceiling is None:
        ceiling = effective_context_ceiling(
            model_path,
            model_format=model_format,
            model_name=model_name,
        )
    if not env_bool("SEISO_LLAMA_DYNAMIC_CTX", True):
        sized = default
    else:
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        est_prompt = max(256, int(chars / 3.2))
        needed = est_prompt + max_tokens + 128
        step = 512
        sized = min(ceiling, max(floor, ((needed + step - 1) // step) * step))
    from seiso.memory.protection import clamp_llama_n_ctx

    return clamp_llama_n_ctx(
        sized,
        messages=messages,
        max_tokens=max_tokens,
        model_path=model_path,
        model_format=model_format,
        model_name=model_name,
    )


def attach_llama_prompt_cache(llm: Any) -> None:
    """Enable RAM prefix cache for multi-turn / repeated prompts."""
    if not env_bool("SEISO_LLAMA_PROMPT_CACHE", True):
        return
    if getattr(llm, "_seiso_cache_attached", False):
        return
    try:
        from llama_cpp import LlamaRAMCache

        cache_mb = env_int("SEISO_LLAMA_CACHE_MB", 1024)
        from seiso.memory.protection import clamp_llama_cache_mb

        cache_mb = clamp_llama_cache_mb(cache_mb)
        if cache_mb <= 0:
            return
        llm.set_cache(LlamaRAMCache(capacity_bytes=cache_mb * 1024 * 1024))
        llm._seiso_cache_attached = True
        logger.debug("llama.cpp RAM prompt cache enabled (%d MB)", cache_mb)
    except Exception as exc:
        logger.debug("llama.cpp prompt cache skipped: %s", exc)


def torch_generate_kwargs(
    payload: dict[str, Any],
    inputs: dict[str, Any],
    streamer: Any,
    *,
    pad_token_id: int | None = None,
) -> dict[str, Any]:
    temperature = float(payload.get("temperature", 0.0))
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": payload.get("max_tokens", 512),
        "streamer": streamer,
        "use_cache": True,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id

    if temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = max(temperature, 0.01)
        top_p = payload.get("top_p")
        if top_p:
            kwargs["top_p"] = float(top_p)
    else:
        kwargs["do_sample"] = False
    return kwargs


def llama_completion_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    temperature = float(payload.get("temperature", 0.0))
    kwargs: dict[str, Any] = {
        "max_tokens": payload.get("max_tokens", 512),
        "stream": True,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        top_p = payload.get("top_p")
        if top_p:
            kwargs["top_p"] = float(top_p)
    else:
        kwargs["temperature"] = 0.0
    return kwargs


def extract_mlx_token_text(token: Any) -> str | None:
    """Decode a chunk from mlx_lm stream_generate."""
    if token is None:
        return None
    if hasattr(token, "text"):
        text = token.text
        return text if text else None
    if isinstance(token, tuple) and token:
        chunk = token[0]
        return str(chunk) if chunk else None
    text = str(token)
    return text if text else None
