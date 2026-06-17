"""Training configuration and trainer dispatch."""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class TrainMethod(enum.StrEnum):
    LORA = "lora"
    FULL = "full"
    EMBEDDING = "embedding"


class QuantMode(enum.StrEnum):
    NONE = "none"
    INT4 = "4bit"
    INT8 = "8bit"
    INT16 = "16bit"


class DatasetFormat(enum.StrEnum):
    AUTO = "auto"
    TEXT = "text"
    ALPACA = "alpaca"
    CHAT = "chat"
    SHAREGPT = "sharegpt"


class TrainConfig(BaseModel):
    model_id: str
    dataset: str | Path
    output_dir: Path = Field(default=Path("./outputs"))
    method: TrainMethod = TrainMethod.LORA
    quant: QuantMode = QuantMode.INT4
    dataset_format: DatasetFormat = DatasetFormat.AUTO
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    max_seq_length: int = Field(default=2048, ge=128)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, le=0.5)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    gradient_checkpointing: bool = True
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    eval_steps: int | None = None
    eval_split_ratio: float = Field(default=0.05, ge=0, le=0.5)
    lr_scheduler: str = "cosine"
    resume_from: Path | None = None
    seed: int = 42
    deterministic: bool = True
    multi_gpu: bool = False
    use_triton: bool = True
    use_fused_ce: bool = True
    use_fused_lora: bool = True
    use_rslora: bool = False
    train_on_responses_only: bool = True
    packing: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("output_dir", "dataset", "resume_from", mode="before")
    @classmethod
    def _expand_path(cls, v: Any) -> Any:
        if v is None:
            return v
        return Path(v).expanduser()

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)


def run_training(config: TrainConfig, *, on_metric=None) -> Path:
    """Execute training job; returns output checkpoint directory."""
    from seiso.models.hf_env import configure_hf_hub_cache
    from seiso.security.nvidia_boundary import enforce_nvidia_secure_boundary
    from seiso.training.trainer import SeisoTrainer

    configure_hf_hub_cache()
    enforce_nvidia_secure_boundary(context="training")
    trainer = SeisoTrainer(config, on_metric=on_metric)
    return trainer.run()
