"""Family-level LoRA target module detection for Hugging Face models."""

from __future__ import annotations

import re
from contextlib import suppress
from typing import Any

from seiso.models.catalog import ModelFamily, infer_model_family

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

_LLAMA_STYLE_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Family → LoRA targets. Most open LLM families share llama-style linear names.
_FAMILY_TARGETS: dict[ModelFamily, list[str]] = {
    ModelFamily.LLAMA: _LLAMA_STYLE_TARGETS,
    ModelFamily.QWEN: _LLAMA_STYLE_TARGETS,
    ModelFamily.GEMMA: _LLAMA_STYLE_TARGETS,
    ModelFamily.MISTRAL: _LLAMA_STYLE_TARGETS,
    ModelFamily.DEEPSEEK: _LLAMA_STYLE_TARGETS,
    ModelFamily.KIMI: _LLAMA_STYLE_TARGETS,
    ModelFamily.GLM: _LLAMA_STYLE_TARGETS,
    ModelFamily.OLMO: _LLAMA_STYLE_TARGETS,
    ModelFamily.GRANITE: _LLAMA_STYLE_TARGETS,
    ModelFamily.YI: _LLAMA_STYLE_TARGETS,
    ModelFamily.INTERNLM: _LLAMA_STYLE_TARGETS,
    ModelFamily.BAICHUAN: _LLAMA_STYLE_TARGETS,
    ModelFamily.STABLELM: _LLAMA_STYLE_TARGETS,
    ModelFamily.COMMAND: _LLAMA_STYLE_TARGETS,
    ModelFamily.SMOLLM: _LLAMA_STYLE_TARGETS,
    ModelFamily.GPT_OSS: _LLAMA_STYLE_TARGETS,
    ModelFamily.EXAONE: _LLAMA_STYLE_TARGETS,
    ModelFamily.NOVA: _LLAMA_STYLE_TARGETS,
}

# Architectures whose module names differ from the llama-style default.
_ARCHITECTURE_TARGETS: dict[str, list[str] | str] = {
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
    "qwen2_moe": _LLAMA_STYLE_TARGETS,
    "qwen3_moe": _LLAMA_STYLE_TARGETS,
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
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

_PHI3_RE = re.compile(r"(^|[-_/])phi[-_]?[34]", re.I)


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


def _special_architecture(model_id: str, model_type: str | None) -> str | None:
    hay = f"{model_id} {model_type or ''}".lower()
    if "mixtral" in hay or model_type == "mixtral":
        return "mixtral"
    if _PHI3_RE.search(hay) or model_type in {"phi3", "phi4"}:
        return "phi3"
    if "phi" in hay or model_type == "phi":
        return "phi"
    if "falcon" in hay or model_type == "falcon":
        return "falcon"
    if model_type in {"gpt2"}:
        return "gpt2"
    if model_type in {"gpt_neox", "gptj"}:
        return "gpt_neox"
    return None


def detect_architecture(model_id: str, model=None) -> str | None:
    """Detect model architecture from Hub id, config, or family classification."""
    model_type = _config_model_type(model) if model is not None else None
    if model_type in _ARCHITECTURE_TARGETS:
        return model_type

    tag_hints = [model_type] if model_type else []
    special = _special_architecture(model_id, model_type)
    if special:
        return special

    family = infer_model_family(model_id, tag_hints)
    if family != ModelFamily.OTHER:
        return family.value

    return model_type


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
    try:
        family = ModelFamily(arch) if arch else ModelFamily.OTHER
    except ValueError:
        family = ModelFamily.OTHER
    if family in _FAMILY_TARGETS:
        return _FAMILY_TARGETS[family]
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
        suffix for suffix in suffixes if suffix.endswith(("proj", "dense", "linear", "fc"))
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

    has_language_model = any(_LANGUAGE_MODEL_MARKER in name.split(".") for name in module_names)
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


def modules_exist_in_model(model, target_modules: list[str] | str) -> list[str] | str:
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
