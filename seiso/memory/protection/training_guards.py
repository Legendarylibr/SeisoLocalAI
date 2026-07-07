"""Training memory guardrails."""

from __future__ import annotations

import logging
from typing import Any

from seiso import platform as seiso_platform
from seiso.env import env_bool
from seiso.hardware import vram_headroom_mb
from seiso.memory.estimates import guess_params_from_name
from seiso.memory.protection._facade import protection

logger = logging.getLogger(__name__)
_MEMORY_POLICY_FIELDS = (
    "batch_size",
    "gradient_accumulation_steps",
    "max_seq_length",
    "quant",
)


def training_pin_memory() -> bool:
    """Pin memory only when CUDA training is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _training_caps_for_model(
    config: Any,
    defaults: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, int]:
    batch = max(1, int(defaults.get("batch_size") or 1))
    accum = max(1, int(defaults.get("gradient_accumulation_steps") or 8))
    max_seq = max(128, int(defaults.get("max_seq_length") or 2048))

    params_b = guess_params_from_name(str(getattr(config, "model_id", "")))
    if params_b is not None:
        if params_b <= 1.0:
            model_batch, model_accum, model_seq = 4, 2, max_seq
        elif params_b <= 3.0:
            model_batch, model_accum, model_seq = 2, 4, max_seq
        elif params_b <= 7.0:
            model_batch, model_accum, model_seq = 1, 8, min(max_seq, 2048)
        elif params_b <= 14.0:
            model_batch, model_accum, model_seq = 1, 16, min(max_seq, 2048)
        else:
            model_batch, model_accum, model_seq = 1, 32, min(max_seq, 1024)
        batch = min(batch, model_batch)
        accum = max(accum, model_accum)
        max_seq = min(max_seq, model_seq)

    try:
        native_linux_nvidia = seiso_platform.use_linux_nvidia_inference_guards()
    except Exception:
        native_linux_nvidia = False
    headroom = vram_headroom_mb(profile)
    if (native_linux_nvidia and headroom <= 24576 and params_b is not None and params_b > 7.0) or (
        headroom > 0 and headroom < 8192
    ):
        max_seq = min(max_seq, 1024)

    return {
        "batch_size": batch,
        "gradient_accumulation_steps": accum,
        "max_seq_length": max_seq,
    }



def apply_training_memory_guards(config: Any) -> Any:
    """Clamp memory-sensitive training knobs to hardware/model-safe ceilings."""
    from seiso.training.config import TrainConfig

    if not isinstance(config, TrainConfig):
        return config

    profile = protection().hardware_profile()
    try:
        defaults = protection().training_defaults(profile)
    except Exception:
        defaults = {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
        }

    updates: dict[str, Any] = {}

    if not env_bool("SEISO_TRAINING_UNSAFE_BATCH", False):
        caps = _training_caps_for_model(config, defaults, profile)
        current_batch = max(1, int(config.batch_size))
        capped_batch = min(current_batch, caps["batch_size"])
        if capped_batch < current_batch:
            updates["batch_size"] = capped_batch
            effective_batch = current_batch * max(1, int(config.gradient_accumulation_steps))
            preserved_accum = max(
                caps["gradient_accumulation_steps"],
                (effective_batch + capped_batch - 1) // capped_batch,
            )
            if preserved_accum > int(config.gradient_accumulation_steps):
                updates["gradient_accumulation_steps"] = preserved_accum
        elif int(config.gradient_accumulation_steps) < caps["gradient_accumulation_steps"]:
            updates["gradient_accumulation_steps"] = caps["gradient_accumulation_steps"]

        capped_seq = min(int(config.max_seq_length), caps["max_seq_length"])
        if capped_seq < int(config.max_seq_length):
            updates["max_seq_length"] = capped_seq

    # Downgrade quant to the platform-recommended value when the requested mode
    # is unavailable (e.g. QLoRA/4-bit on macOS where bitsandbytes is absent,
    # or 4-bit on a CPU-only box). Without this, torch_loader silently loads a
    # 16-bit model while the trainer still requests paged_adamw_8bit, crashing
    # at optimizer creation with ImportError: bitsandbytes.
    recommended_quant = defaults.get("quant")
    if recommended_quant and str(config.quant) != str(recommended_quant):
        target: Any = None
        try:
            from seiso.training.config import QuantMode

            target = QuantMode(recommended_quant)
        except (ValueError, ImportError):
            target = None
        if target is not None and target != config.quant:
            # Only downgrade — never upgrade beyond what the user asked for.
            rank = {
                QuantMode.NONE: 0,
                QuantMode.INT16: 1,
                QuantMode.INT8: 2,
                QuantMode.INT4: 3,
            }  # type: ignore[name-defined]
            if rank.get(target, 0) < rank.get(config.quant, 0):
                updates["quant"] = target
                logger.info(
                    "Training memory guards: quant %s -> %s (platform recommendation)",
                    config.quant.value,
                    target.value,
                )

    if not updates:
        return config

    logger.info("Training memory guards applied: %s", updates)
    return config.model_copy(update=updates)


def describe_training_memory_policy(
    original: Any,
    guarded: Any,
    *,
    reason: str,
) -> dict[str, Any]:
    """Return user-visible details about guard/fallback changes."""
    changes: dict[str, dict[str, Any]] = {}
    for field in _MEMORY_POLICY_FIELDS:
        before = getattr(original, field, None)
        after = getattr(guarded, field, None)
        before_value = getattr(before, "value", before)
        after_value = getattr(after, "value", after)
        if before_value != after_value:
            changes[field] = {"from": before_value, "to": after_value}
    return {
        "reason": reason,
        "changed": bool(changes),
        "changes": changes,
    }


def apply_training_oom_fallback(config: Any) -> Any:
    """Halve batch / seq after an OOM during training."""
    batch = max(1, int(config.batch_size) // 2)
    accum = int(config.gradient_accumulation_steps) * 2
    max_seq = max(128, int(config.max_seq_length) // 2)
    logger.warning(
        "OOM recovery: batch_size=%d accum=%d max_seq_length=%d",
        batch,
        accum,
        max_seq,
    )
    return config.model_copy(
        update={
            "batch_size": batch,
            "gradient_accumulation_steps": accum,
            "max_seq_length": max_seq,
        }
    )

