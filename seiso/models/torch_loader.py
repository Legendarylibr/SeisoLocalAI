"""PyTorch/CUDA/MPS model loader with optional quantization."""

from __future__ import annotations

import logging
import platform
from typing import Any

from seiso.models.loader import Backend, LoadOptions

logger = logging.getLogger(__name__)


def _resolve_dtype(options: LoadOptions) -> Any:
    if not options.dtype:
        return None
    import torch

    return getattr(torch, options.dtype, None)


def _resolve_device_map(
    backend: Backend,
    device: str | None = None,
    *,
    for_training: bool = False,
) -> str | dict[str, str] | None:
    if for_training:
        from seiso.memory.protection import resolve_training_device_map

        return resolve_training_device_map(device)

    import torch

    if device == "mps" or (
        device is None
        and backend != Backend.TORCH
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return {"": "mps"}
    if device == "cuda" or (backend == Backend.TORCH and torch.cuda.is_available()):
        return "auto"
    return None


def load_torch(
    options: LoadOptions,
    backend: Backend = Backend.TORCH,
    *,
    device: str | None = None,
    for_training: bool = False,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": options.trust_remote_code,
        "revision": options.revision or "main",
    }
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": options.trust_remote_code,
        "revision": options.revision or "main",
    }

    device_map = _resolve_device_map(backend, device, for_training=for_training)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
        if device_map == "auto":
            from seiso.memory.protection import build_hf_max_memory

            max_memory = build_hf_max_memory()
            if max_memory:
                model_kwargs["max_memory"] = max_memory

    dtype = _resolve_dtype(options)
    if dtype is None and device != "mps":
        try:
            import torch

            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except ImportError:
            dtype = None
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model_kwargs.setdefault("low_cpu_mem_usage", True)

    if options.use_flash_attention and device != "mps":
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info("Using SDPA attention")

    use_4bit = options.load_in_4bit
    use_8bit = options.load_in_8bit
    if platform.system() == "Darwin" and (use_4bit or use_8bit):
        logger.warning("bitsandbytes QLoRA is unavailable on macOS — loading without quantization")
        use_4bit = False
        use_8bit = False

    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_kwargs.get("torch_dtype"),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        model_kwargs["load_in_8bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(options.model_id, **tokenizer_kwargs)  # nosec B615: revision pinned in tokenizer_kwargs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(options.model_id, **model_kwargs)  # nosec B615: revision pinned in model_kwargs

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
