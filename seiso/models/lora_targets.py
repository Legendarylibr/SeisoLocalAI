"""Model-specific LoRA target module detection."""

from __future__ import annotations

import re

# Architecture patterns → LoRA target modules
_ARCHITECTURE_TARGETS: dict[str, list[str]] = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma4": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "yi": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

_DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def detect_architecture(model_id: str, model=None) -> str:
    """Detect model architecture from ID or config."""
    mid = model_id.lower()
    patterns = [
        (r"llama|tinyllama", "llama"),
        (r"mixtral", "mixtral"),
        (r"mistral|codestral", "mistral"),
        (r"qwen3", "qwen3"),
        (r"qwen2|qwen", "qwen2"),
        (r"gemma-4|gemma4", "gemma4"),
        (r"gemma-2|gemma2", "gemma2"),
        (r"gemma-3|gemma3", "gemma"),
        (r"gemma", "gemma"),
        (r"phi-3|phi3|phi-4|phi4", "phi3"),
        (r"phi", "phi"),
        (r"deepseek", "deepseek"),
        (r"falcon", "falcon"),
        (r"yi-", "yi"),
        (r"gpt-oss", "llama"),
    ]
    for pat, arch in patterns:
        if re.search(pat, mid):
            return arch

    if model is not None:
        cfg = getattr(model, "config", None)
        if cfg:
            mt = getattr(cfg, "model_type", "").lower()
            if mt in _ARCHITECTURE_TARGETS:
                return mt

    return "llama"


def get_lora_target_modules(model_id: str, model=None) -> list[str]:
    """Return LoRA target modules for a given model."""
    arch = detect_architecture(model_id, model)
    return _ARCHITECTURE_TARGETS.get(arch, _DEFAULT_TARGETS)


def modules_exist_in_model(model, target_modules: list[str]) -> list[str]:
    """Filter target modules to those actually present in the model."""
    param_names = {n for n, _ in model.named_parameters()}
    found: set[str] = set()
    for name in param_names:
        for t in target_modules:
            if name.endswith(t) or f".{t}." in name or name.endswith(f".{t}"):
                found.add(t)
    return list(found) if found else target_modules
