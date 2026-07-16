"""Slime-style RL training (local HF, multi-GPU DDP, SGLang/vLLM rollouts).

Historical import path ``seiso.slime_single_gpu`` remains as a compatibility
shim. Prefer ``seiso.slime`` for new code.

Public names:

- ``SlimeConfig`` / ``SingleGpuSlimeConfig`` (alias)
- ``train_slime`` / ``train_single_gpu_slime`` (alias; import from ``.trainer``)
"""

from __future__ import annotations

from seiso.slime.config import SingleGpuSlimeConfig, SlimeConfig

__all__ = ["SlimeConfig", "SingleGpuSlimeConfig"]
