"""RL quant memory guardrails."""

from __future__ import annotations

from typing import Any

from seiso.env import env_bool
from seiso.memory.protection._facade import protection
from seiso.memory.protection.llama_clamp import clamp_llama_n_ctx


def _rl_batch_caps_for_headroom(free_mb: int) -> dict[str, int]:
    if free_mb <= 0:
        return {}
    if free_mb < 4096:
        return {
            "torch_preflight_batch_size": 512,
            "torch_batch_episodes": 128,
            "torch_minibatch_size": 64,
            "online_batch_size": 32,
            "continuous_batch_size": 16,
        }
    if free_mb < 8192:
        return {
            "torch_preflight_batch_size": 1024,
            "torch_batch_episodes": 256,
            "torch_minibatch_size": 128,
            "online_batch_size": 64,
            "continuous_batch_size": 32,
        }
    if free_mb < 16384:
        return {
            "torch_preflight_batch_size": 2048,
            "torch_batch_episodes": 512,
            "torch_minibatch_size": 256,
            "online_batch_size": 96,
            "continuous_batch_size": 48,
        }
    return {
        "torch_preflight_batch_size": 4096,
        "torch_batch_episodes": 1024,
        "torch_minibatch_size": 512,
        "online_batch_size": 128,
        "continuous_batch_size": 64,
    }


def apply_rl_memory_guards(flat: dict[str, Any]) -> dict[str, Any]:
    """Clamp RL quant batch-like knobs before torch allocates large tensors."""
    out = dict(flat)
    ctx = int(out.get("llama_cpp_context") or 0)
    if ctx > 0:
        out["llama_cpp_context"] = min(ctx, clamp_llama_n_ctx(ctx, max_tokens=512))

    if not env_bool("SEISO_RL_UNSAFE_BATCH", False):
        for key, cap in _rl_batch_caps_for_headroom(protection().headroom_mb()).items():
            value = out.get(key)
            if value is not None:
                out[key] = min(int(value), cap)
        if (
            out.get("torch_minibatch_size") is not None
            and out.get("torch_batch_episodes") is not None
        ):
            out["torch_minibatch_size"] = min(
                int(out["torch_minibatch_size"]),
                int(out["torch_batch_episodes"]),
            )

    return out


