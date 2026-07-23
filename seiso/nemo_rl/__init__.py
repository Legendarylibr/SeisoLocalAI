"""NVIDIA NeMo RL integration — optional external GRPO/DPO post-training.

NeMo RL is not vendored. Point ``SEISO_NEMO_RL_ROOT`` (or ``nemo_rl_root``) at a
clone of https://github.com/NVIDIA-NeMo/RL and launch via ``uv run``.
"""

from __future__ import annotations

from seiso.nemo_rl.bootstrap import (
    nemo_rl_available,
    resolve_nemo_rl_root,
    resolve_uv_executable,
)
from seiso.nemo_rl.config import NeMoRLConfig, NeMoRLRecipe
from seiso.nemo_rl.runner import train_nemo_rl

__all__ = [
    "NeMoRLConfig",
    "NeMoRLRecipe",
    "nemo_rl_available",
    "resolve_nemo_rl_root",
    "resolve_uv_executable",
    "train_nemo_rl",
]
