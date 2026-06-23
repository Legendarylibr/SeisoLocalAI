"""Training configuration and trainer dispatch."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from seiso.compat import StrEnum


class TrainMethod(StrEnum):
    LORA = "lora"
    FULL = "full"
    EMBEDDING = "embedding"


class QuantMode(StrEnum):
    NONE = "none"
    INT4 = "4bit"
    INT8 = "8bit"
    INT16 = "16bit"


class DatasetFormat(StrEnum):
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
    sandbox_root: Path | None = None
    seed: int = 42
    deterministic: bool = True
    multi_gpu: bool = False
    use_triton: bool = True
    use_fused_ce: bool = True
    use_fused_lora: bool = True
    use_rslora: bool = False
    train_on_responses_only: bool = True
    packing: bool = False
    # ── Performance optimizations (auto-tuned when left at defaults) ──
    dataloader_num_workers: int = 0  # 0 = auto-detect (min(4, cpu_count//2) on CUDA, 0 on CPU)
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int | None = None  # None = let HF pick (2)
    group_by_length: bool = True  # batch similar-length sequences → less padding waste
    padding_free: bool = False  # use flash-attention padding-free packing (CUDA only)
    neftune_noise_alpha: float | None = 5.0  # NEFTune instruction-tuning noise (None to disable)
    torch_compile: bool = False  # torch.compile the training model (CUDA only, opt-in)
    save_safetensors: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("output_dir", "dataset", "resume_from", "sandbox_root", mode="before")
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


def run_training(
    config: TrainConfig,
    *,
    on_metric=None,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Execute training job; returns output checkpoint directory."""
    from seiso.env import configure_transformers_env
    from seiso.memory.protection import apply_training_memory_guards
    from seiso.models.hf_env import configure_hf_hub_cache
    from seiso.security.nvidia_boundary import enforce_nvidia_secure_boundary
    from seiso.training.trainer import SeisoTrainer

    from seiso.platform import ensure_cuda_library_path

    configure_transformers_env()
    if not os.environ.get("HF_HOME"):
        configure_hf_hub_cache(config.sandbox_root)
    ensure_cuda_library_path()
    enforce_nvidia_secure_boundary(context="training")
    config = apply_training_memory_guards(config)
    trainer = SeisoTrainer(config, on_metric=on_metric, on_log=on_log)
    return trainer.run()
