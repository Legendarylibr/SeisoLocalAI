"""Helpers for Hugging Face models that ship with native hub quantization (MXFP4, FP8)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from seiso.memory.estimates import _active_params_b, guess_params_from_name

logger = logging.getLogger(__name__)

NATIVE_HUB_QUANT_METHODS = frozenset({"mxfp4", "fp8"})
UNTRAINABLE_NATIVE_QUANT_METHODS = frozenset({"fp8"})

NATIVE_QUANT_TRAINING_MESSAGE = (
    "This model ships with native {method} weights on the Hub. Transformers does not "
    "support fine-tuning {method_upper} checkpoints — choose the BF16/FP16 safetensors "
    "variant (without the -FP8 suffix) or pick a smaller trainable base model."
)


def is_hub_model_id(model_id: str) -> bool:
    """True when ``model_id`` looks like a Hugging Face repo id (org/name)."""
    text = str(model_id).strip().replace("\\", "/")
    if text.startswith(("/", "./", "../", "~")):
        return False
    parts = [part for part in text.split("/") if part]
    return len(parts) == 2 and all(part for part in parts)


def native_quant_method_from_config(config: Any) -> str | None:
    """Return quant method from a transformers PretrainedConfig, if native hub quant."""
    pre_quant = getattr(config, "quantization_config", None)
    method = ""
    if isinstance(pre_quant, dict):
        method = str(pre_quant.get("quant_method") or "").lower()
    elif pre_quant is not None:
        method = str(getattr(pre_quant, "quant_method", "") or "").lower()
    return method if method in NATIVE_HUB_QUANT_METHODS else None


@lru_cache(maxsize=32)
def peek_model_config(model_ref: str, trust_remote_code: bool = False) -> Any | None:
    """Load ``AutoConfig`` only — no weights — for a hub id or local snapshot path."""
    ref = str(model_ref).strip()
    if not ref:
        return None
    try:
        from pathlib import Path

        from transformers import AutoConfig

        from seiso.models.hf_env import _read_hub_token

        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        token = _read_hub_token()
        if token:
            kwargs["token"] = token
        path = Path(ref).expanduser()
        if path.exists():
            return AutoConfig.from_pretrained(str(path), **kwargs)
        if is_hub_model_id(ref):
            return AutoConfig.from_pretrained(ref, **kwargs)
    except Exception as exc:
        logger.debug("Could not peek model config for %s: %s", model_ref, exc)
    return None


def peek_hub_config(model_id: str, trust_remote_code: bool = False) -> Any | None:
    """Load ``AutoConfig`` only — no weights — for hub-side quant/arch metadata."""
    if not is_hub_model_id(model_id):
        return None
    return peek_model_config(model_id, trust_remote_code=trust_remote_code)


def is_native_hub_quant_model(
    model_id: str,
    *,
    config: Any | None = None,
    trust_remote_code: bool = False,
    peek: bool = True,
) -> bool:
    """True when the model uses native hub quant weights (not bitsandbytes QLoRA)."""
    if config is not None and native_quant_method_from_config(config) is not None:
        return True
    if peek and config is None:
        config = peek_model_config(model_id, trust_remote_code=trust_remote_code)
        if config is not None and native_quant_method_from_config(config) is not None:
            return True
    return False


def native_quant_training_block_reason(
    model_ref: str,
    *,
    config: Any | None = None,
    trust_remote_code: bool = False,
) -> str | None:
    """Return a user-facing error when native hub quant weights cannot be fine-tuned."""
    method = native_quant_method_from_config(config) if config is not None else None
    if method is None:
        config = config or peek_model_config(
            model_ref, trust_remote_code=trust_remote_code
        )
        method = native_quant_method_from_config(config) if config is not None else None
    if method is None and "-fp8" in str(model_ref).lower():
        method = "fp8"
    if method not in UNTRAINABLE_NATIVE_QUANT_METHODS:
        return None
    return NATIVE_QUANT_TRAINING_MESSAGE.format(
        method=method, method_upper=method.upper()
    )


def active_params_from_config(config: Any) -> float | None:
    """Best-effort active parameter count (billions) from model config."""
    for attr in ("active_parameter_count", "num_active_parameters"):
        value = getattr(config, attr, None)
        if value is not None:
            try:
                count = float(value)
            except (TypeError, ValueError):
                continue
            if count > 1_000:
                return count / 1e9
            if count > 0:
                return count

    total_params = getattr(config, "num_parameters", None)
    num_experts = getattr(config, "num_local_experts", None) or getattr(
        config, "num_experts", None
    )
    experts_per_tok = (
        getattr(config, "num_experts_per_tok", None)
        or getattr(config, "num_selected_experts", None)
        or getattr(config, "num_activated_experts", None)
    )
    if total_params and num_experts and experts_per_tok:
        try:
            active = float(total_params) / float(num_experts) * float(experts_per_tok)
            if active > 0:
                return active / 1e9
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    model_type = str(getattr(config, "model_type", "") or "").lower()
    tags: tuple[str, ...] = ("moe",) if "moe" in model_type else ()
    label = str(
        getattr(config, "name_or_path", "")
        or getattr(config, "_name_or_path", "")
        or ""
    )
    if not label:
        label = f"{getattr(config, 'hidden_size', '')}-{model_type}".strip("-")
    if label:
        return _active_params_b(label, tags, repo_id=label)
    return None


def infer_active_params_b(
    model_id: str,
    *,
    config: Any | None = None,
    trust_remote_code: bool = False,
) -> float:
    """Best-effort active parameter count (billions) for VRAM and regression estimates."""
    if config is None and is_hub_model_id(model_id):
        config = peek_hub_config(model_id, trust_remote_code=trust_remote_code)
    if config is not None:
        from_config = active_params_from_config(config)
        if from_config is not None:
            return from_config

    guessed = guess_params_from_name(model_id)
    label = f"{guessed}B" if guessed is not None else str(model_id).split("/")[-1]
    model_type = str(getattr(config, "model_type", "") or "").lower() if config else ""
    tags: tuple[str, ...] = ("moe",) if "moe" in model_type else ()
    return _active_params_b(label, tags, repo_id=str(model_id))


def needs_tight_vram_training(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    est_train_mb: int = 0,
    headroom_mb: int = 0,
) -> bool:
    """True when training should use conservative batch/seq/kernel settings."""
    if is_native_hub_quant_model(
        model_id,
        trust_remote_code=trust_remote_code,
        peek=True,
    ):
        return True
    if headroom_mb > 0 and est_train_mb > int(headroom_mb * 0.97):
        return True
    params_b = infer_active_params_b(model_id, trust_remote_code=trust_remote_code)
    return params_b >= 14.0
