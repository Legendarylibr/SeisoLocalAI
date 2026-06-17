"""PyTorch/CUDA model loader with optional quantization."""

from __future__ import annotations

import logging
from typing import Any

from seiso.models.loader import Backend, LoadOptions

logger = logging.getLogger(__name__)


def load_torch(options: LoadOptions, backend: Backend = Backend.TORCH) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": options.trust_remote_code}
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": options.trust_remote_code,
        "device_map": options.device_map if backend == Backend.TORCH else None,
    }

    if options.dtype:
        import torch

        model_kwargs["torch_dtype"] = getattr(torch, options.dtype, None)

    if options.use_flash_attention:
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except ImportError:
            pass

    if options.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_kwargs.get("torch_dtype"),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif options.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(options.model_id, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(options.model_id, **model_kwargs)

    # Resize embeddings if tokenizer vocab differs
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
