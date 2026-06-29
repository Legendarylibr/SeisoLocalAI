"""Configuration for the single-GPU slime-style trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SingleGpuSlimeConfig:
    """Small, explicit config for local GRPO-style training."""

    model_id: str
    dataset: Path
    output_dir: Path
    prompt_field: str = "prompt"
    answer_field: str = "answer"
    reward: str = "exact_match"
    reward_field: str = "reward"
    max_vram_gb: float | None = None
    max_prompt_tokens: int = 512
    max_new_tokens: int = 256
    rollouts_per_prompt: int = 4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    epochs: int = 1
    max_steps: int | None = None
    kl_coef: float = 0.0
    clip_ratio: float = 0.2
    temperature: float = 0.9
    top_p: float = 0.95
    seed: int = 17
    dtype: str = "auto"
    device: str = "cuda"
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = False
    trust_remote_code: bool = False
    save_every_steps: int = 100
    log_every_steps: int = 1

    @classmethod
    def from_yaml(cls, path: Path) -> SingleGpuSlimeConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("slime config must be a mapping")
        return cls(
            **{
                key: Path(value) if key in {"dataset", "output_dir"} else value
                for key, value in data.items()
            }
        )

    def validate(self) -> None:
        if self.rollouts_per_prompt < 2:
            raise ValueError("rollouts_per_prompt must be at least 2 for grouped advantages")
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_prompt_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("token limits must be positive")
        if self.kl_coef < 0:
            raise ValueError("kl_coef must be non-negative")
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive")
        if self.max_vram_gb is not None and self.max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be positive")
