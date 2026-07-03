"""Model-specific LoRA target module detection."""

from __future__ import annotations

import re
from contextlib import suppress
from typing import Any

# PEFT accepts a regex target. Use it for multimodal wrappers so LoRA skips
# vision/audio towers and only hits the text backbone under language_model.
_MULTIMODAL_LANGUAGE_MODEL_LORA_TARGET_REGEX = r".*language_model\..*\.(q_proj|v_proj)"
_LANGUAGE_MODEL_MARKER = "language_model"
_MULTIMODAL_BACKBONE_MARKERS = frozenset(
    {
        "vision_tower",
        "audio_tower",
        "vision_model",
        "audio_model",
        "visual",
        "multi_modal_projector",
        "mm_projector",
        "embed_vision",
        "embed_audio",
    }
)

# Architecture patterns → LoRA target modules (regex for special multimodal cases)
_ARCHITECTURE_TARGETS: dict[str, list[str] | str] = {
    "llama": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "mistral": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
    "qwen": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen2": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen3": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "gemma3": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "gemma": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "gemma2": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gpt_oss": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "deepseek": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "yi": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

_DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

_COMMON_LINEAR_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query_key_value",
    "qkv_proj",
    "W_pack",
    "in_proj",
    "out_proj",
    "c_attn",
    "c_proj",
    "c_fc",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
    "fc1",
    "fc2",
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
    "w1",
    "w2",
    "w3",
)


def _config_model_type(model: Any) -> str | None:
    cfg = getattr(model, "config", None)
    if cfg:
        mt = getattr(cfg, "model_type", "")
        if isinstance(mt, str) and mt.strip():
            return mt.lower()
        architectures = getattr(cfg, "architectures", None)
        if isinstance(architectures, (list, tuple)):
            for arch in architectures:
                if isinstance(arch, str) and arch.strip():
                    return arch.lower()
    return None


def detect_architecture(model_id: str, model=None) -> str | None:
    """Detect model architecture from ID or config."""
    if model is not None:
        mt = _config_model_type(model)
        if mt in _ARCHITECTURE_TARGETS:
            return mt

    mid = model_id.lower()
    patterns = [
        (r"llama|tinyllama", "llama"),
        (r"mixtral", "mixtral"),
        (r"mistral|codestral|magistral", "mistral"),
        (r"qwq|qwen3\.5|qwen3", "qwen3"),
        (r"qwen2|qwen", "qwen2"),
        (r"gemma-3|gemma3", "gemma3"),
        (r"gemma-2|gemma2", "gemma2"),
        (r"gemma", "gemma"),
        (r"phi-3|phi3|phi-4|phi4", "phi3"),
        (r"phi", "phi"),
        (r"deepseek-r1|deepseek_r1|deepseek", "deepseek"),
        (r"falcon", "falcon"),
        (r"yi-", "yi"),
        (r"gpt-oss|gpt_oss", "gpt_oss"),
    ]
    for pat, arch in patterns:
        if re.search(pat, mid):
            return arch

    if model is not None:
        return _config_model_type(model)

    return None


def _target_suffix(name: str) -> str:
    suffix = name.rsplit(".", 1)[-1]
    if suffix == "weight":
        parts = name.split(".")
        if len(parts) >= 2:
            return parts[-2]
    return suffix


def get_lora_target_modules(model_id: str, model=None) -> list[str] | str:
    """Return LoRA target modules for a given model."""
    if model is not None and has_multimodal_language_model_backbone(model):
        return _MULTIMODAL_LANGUAGE_MODEL_LORA_TARGET_REGEX
    arch = detect_architecture(model_id, model)
    if arch in _ARCHITECTURE_TARGETS:
        return _ARCHITECTURE_TARGETS[arch]
    if model is not None:
        inferred = infer_lora_target_modules(model)
        if inferred:
            return inferred
    return _DEFAULT_TARGETS


def infer_lora_target_modules(model) -> list[str]:
    """Infer likely LoRA targets from the model's actual parameter names."""
    try:
        param_names = [n for n, _ in model.named_parameters()]
    except Exception:
        return []

    suffixes = {_target_suffix(name) for name in param_names}
    targets = [target for target in _COMMON_LINEAR_TARGETS if target in suffixes]
    if targets:
        return targets

    generic = sorted(
        suffix
        for suffix in suffixes
        if suffix.endswith(("proj", "dense", "linear", "fc"))
    )
    return generic


def has_multimodal_language_model_backbone(model) -> bool:
    """Return true for multimodal wrappers with a trainable text LM subtree."""
    try:
        module_names = [name for name, _ in model.named_modules() if name]
    except Exception:
        module_names = []

    if not module_names:
        with suppress(Exception):
            module_names = [name for name, _ in model.named_parameters()]

    has_language_model = any(
        _LANGUAGE_MODEL_MARKER in name.split(".") for name in module_names
    )
    has_multimodal_backbone = any(
        any(part in _MULTIMODAL_BACKBONE_MARKERS for part in name.split("."))
        for name in module_names
    )
    return has_language_model and has_multimodal_backbone


_LINEAR_MODULE_TYPES = frozenset({"linear", "conv1d"})


def infer_lora_target_modules_from_module_tree(model) -> list[str]:
    """Infer LoRA targets from module names when parameter naming is opaque."""
    try:
        modules = list(model.named_modules())
    except Exception:
        return []

    tails = {name.rsplit(".", 1)[-1] for name, _module in modules if name}
    preferred = [target for target in _COMMON_LINEAR_TARGETS if target in tails]
    if preferred:
        return preferred

    return sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in modules
            if name
            and name.rsplit(".", 1)[-1] != "lm_head"
            and module.__class__.__name__.lower() in _LINEAR_MODULE_TYPES
        }
    )


def resolve_lora_target_modules(
    model_id: str,
    model,
    configured: list[str] | None = None,
) -> list[str] | str:
    """Pick LoRA targets for any model: explicit config → arch map → param → module tree."""
    if configured is not None:
        unique_configured = list(dict.fromkeys(configured))
        resolved = modules_exist_in_model(model, unique_configured)
        if resolved:
            return resolved

    for candidates in (
        get_lora_target_modules(model_id, model),
        infer_lora_target_modules(model),
        infer_lora_target_modules_from_module_tree(model),
    ):
        if not candidates:
            continue
        resolved = modules_exist_in_model(model, candidates)
        if resolved:
            return resolved

    raise ValueError(
        "Could not infer LoRA target modules for this model. "
        "Pass lora_target_modules explicitly for this architecture."
    )


def modules_exist_in_model(
    model, target_modules: list[str] | str
) -> list[str] | str:
    """Filter target modules to those actually present in the model."""
    if isinstance(target_modules, str):
        try:
            module_names = [name for name, _ in model.named_modules() if name]
        except Exception:
            return []
        if any(re.fullmatch(target_modules, name) for name in module_names):
            return target_modules
        return []

    names: set[str] = set()
    with suppress(Exception):
        names.update(name for name, _ in model.named_parameters())
    with suppress(Exception):
        names.update(name for name, _ in model.named_modules() if name)
    found: set[str] = set()
    for name in names:
        for t in target_modules:
            if name.endswith(t) or f".{t}." in name or name.endswith(f".{t}"):
                found.add(t)
    return [t for t in target_modules if t in found]
