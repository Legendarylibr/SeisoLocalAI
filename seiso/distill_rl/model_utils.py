"""Shared model load/release helpers for distill-RL stages."""

from __future__ import annotations

import gc
from typing import Any

import torch


def release_causal_lm(model: Any) -> None:
    """Release a causal LM and cached GPU memory."""
    del model
    gc.collect()
    try:
        from seiso.memory.protection import release_cached_memory

        release_cached_memory(sync=True)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_causal_lm(
    model_path: str,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision:
        load_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_dtype = (
        dtype
        if dtype is not None
        else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        **load_kwargs,
    }
    if model_kwargs["device_map"] == "auto":
        from seiso.memory.protection import build_hf_max_memory

        max_memory = build_hf_max_memory()
        if max_memory:
            model_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not torch.cuda.is_available():
        model = model.to(device="cpu")  # type: ignore[call-arg]
    model.eval()
    device = getattr(model, "device", next(model.parameters()).device)
    return model, tokenizer, device
