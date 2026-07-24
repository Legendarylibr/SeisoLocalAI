"""Training configuration and trainer dispatch."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from seiso.compat import StrEnum


class TrainMethod(StrEnum):
    LORA = "lora"
    FULL = "full"
    EMBEDDING = "embedding"
    SLIME = "slime"
    NEMO_RL = "nemo_rl"


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
    PREFERENCE = "preference"


class DistributedStrategy(StrEnum):
    AUTO = "auto"
    NONE = "none"
    DDP = "ddp"


class CloudGpuProvider(StrEnum):
    NONE = "none"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LAMBDA = "lambda"
    RUNPOD = "runpod"
    COREWEAVE = "coreweave"
    CUSTOM = "custom"


class TrainConfig(BaseModel):
    model_id: str
    dataset: str | Path
    output_dir: Path = Field(default=Path("./outputs"))
    method: TrainMethod = TrainMethod.LORA
    quant: QuantMode = QuantMode.INT4
    dataset_format: DatasetFormat = DatasetFormat.AUTO
    epochs: int = Field(
        default=3,
        ge=1,
        description="Maximum training epochs (early stopping may finish earlier)",
    )
    batch_size: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    max_seq_length: int = Field(default=2048, ge=128)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, le=0.5)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    gradient_checkpointing: bool = True
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    eval_steps: int | None = None
    eval_split_ratio: float = Field(default=0.05, ge=0, le=0.5)
    max_eval_samples: int = Field(
        default=128,
        ge=1,
        description="Cap validation rows so most of the dataset stays in training",
    )
    preprocess_dataset: bool = True
    deduplicate_dataset: bool = True
    min_sample_chars: int = Field(default=1, ge=0)
    early_stopping: bool = True
    early_stopping_patience: int = Field(default=3, ge=1)
    early_stopping_threshold: float = Field(default=0.001, ge=0)
    metric_for_best_model: str = "eval_loss"
    lr_scheduler: str = "cosine"
    resume_from: Path | None = None
    sandbox_root: Path | None = None
    #: When set with sandbox_root=data_dir, enforce per-user scoped roots.
    sandbox_user_id: str | None = None
    seed: int = 42
    deterministic: bool = True
    multi_gpu: bool = False
    distributed_strategy: DistributedStrategy = Field(
        default=DistributedStrategy.AUTO,
        description="High-level distributed training strategy (auto, none, ddp).",
    )
    distributed_nproc_per_node: int | None = Field(
        default=None,
        ge=1,
        description="Local Accelerate worker count (None = all visible training GPUs).",
    )
    distributed_num_nodes: int = Field(
        default=1,
        ge=1,
        description="Total machines for Accelerate distributed training.",
    )
    distributed_node_rank: int = Field(
        default=0,
        ge=0,
        description="Rank of this machine for multi-machine Accelerate launches.",
    )
    distributed_master_addr: str = Field(
        default="127.0.0.1",
        min_length=1,
        description="Accelerate rendezvous address for multi-machine launches.",
    )
    distributed_master_port: int = Field(
        default=29500,
        ge=1,
        le=65535,
        description="Accelerate rendezvous port.",
    )
    ddp_backend: str | None = Field(
        default=None,
        description="Optional HuggingFace/PyTorch DDP backend (for example nccl or gloo).",
    )
    ddp_find_unused_parameters: bool = False
    cloud_gpu_enabled: bool = False
    cloud_gpu_provider: CloudGpuProvider = CloudGpuProvider.NONE
    cloud_gpu_region: str = ""
    cloud_gpu_instance_type: str = ""
    cloud_gpu_count: int | None = Field(default=None, ge=1, le=1024)
    cloud_gpu_project: str = ""
    cloud_gpu_credential_id: str | None = Field(
        default=None,
        description="Encrypted cloud GPU credential record to use for external launchers.",
    )
    use_triton: bool = True
    use_fused_ce: bool = True
    use_fused_lora: bool = True
    use_rslora: bool = False
    train_on_responses_only: bool = True
    packing: bool = False
    preference_as_sft: bool = Field(
        default=False,
        description=(
            "When true, preference (chosen/rejected) rows train SFT on chosen only. "
            "Default false refuses preference data — use Distill-RL/DPO for real alignment."
        ),
    )
    assistant_only_loss: bool | None = Field(
        default=None,
        description="TRL assistant-only loss when the trainer tokenizes chat rows (None = auto)",
    )
    dataset_num_proc: int | None = Field(
        default=None,
        description="Parallel workers for dataset map/tokenize (None = auto, 0 = disable)",
    )
    pad_to_multiple_of: int | None = Field(
        default=None,
        description="Pad batch sequences to this multiple for tensor cores (None = 8 on CUDA)",
    )
    # ── Performance optimizations (auto-tuned when left at defaults) ──
    dataloader_num_workers: int = 0  # 0 = auto-detect (min(4, cpu_count//2) on CUDA, 0 on CPU)
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int | None = None  # None = auto 2 when CUDA workers are enabled
    group_by_length: bool = True  # batch similar-length sequences → less padding waste
    padding_free: bool = False  # use flash-attention padding-free packing (CUDA only)
    neftune_noise_alpha: float | None = 5.0  # NEFTune instruction-tuning noise (None to disable)
    torch_compile: bool = False  # torch.compile the training model (CUDA only, opt-in)
    save_safetensors: bool = True
    training_methodology: str = Field(
        default="seiso_release_post_training",
        description="Stable methodology label written into manifests and snapshots.",
    )
    prompt_field: str = "prompt"
    answer_field: str = "label"
    metadata_field: str | None = "metadata"
    reward: str = "auto"
    reward_field: str = "reward"
    max_vram_gb: float | None = Field(default=None, gt=0)
    max_prompt_tokens: int = Field(default=512, ge=1)
    max_new_tokens: int = Field(default=256, ge=1)
    rollouts_per_prompt: int = Field(default=4, ge=2)
    # slime --rollout-batch-size (prompts)
    rollout_batch_size: int = Field(default=1, ge=1)
    over_sampling_batch_size: int | None = Field(default=None, ge=1)
    dynamic_sampling_filter: str = "reward_nonzero_std"
    dynamic_sampling_min_reward_std: float = Field(default=1e-6, ge=0)
    policy_micro_batch_size: int | None = Field(default=None, ge=1)
    # None → same as rollout_batch_size (slime)
    train_batch_size: int | None = Field(default=None, ge=1)
    balance_data: bool = False
    shuffle_buffer_size: int = Field(default=2048, ge=1)
    max_samples_per_epoch: int | None = Field(default=None, ge=1)
    kl_coef: float = Field(default=0.0, ge=0)
    clip_ratio: float = Field(default=0.2, gt=0, lt=1)
    clip_ratio_high: float | None = Field(
        default=None,
        gt=0,
        description="Upper PPO clip bound (slime eps_clip_high); None uses clip_ratio.",
    )
    clip_ratio_c: float | None = Field(
        default=3.0,
        description=(
            "Dual-clip constant for negative advantages (OpenRLHF/verl/slime); "
            "None disables. Must be > 1 when set."
        ),
    )
    grpo_std_normalization: bool = Field(
        default=True,
        description="Group-relative / unbiased-std advantages (slime grpo_std_normalization).",
    )
    calculate_per_token_loss: bool = Field(
        default=True,
        description=(
            "Slime GRPO: per-token clipped surrogate (length-stable). "
            "When false, sequence log-probs are length-normalized before the ratio."
        ),
    )
    loss_aggregation: str = Field(
        default="seq_mean",
        description=(
            "GRPO loss reduction: seq_mean (DeepSeekMath) or token_mean (length-biased)."
        ),
    )
    temperature: float = Field(default=0.9, gt=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    rollout_backend: str = Field(
        default="hf",
        description=(
            "slime online generate: hf | sglang | vllm | auto. "
            "hf is colocated/on-policy. sglang/vllm sample remotely; Seiso then "
            "recomputes old_logprobs on the local actor (engine sampling logprobs "
            "are not used). Weight sync is required for HTTP backends so engines "
            "cannot drift and bias GRPO importance ratios."
        ),
    )
    apply_chat_template: bool = True
    sglang_base_url: str = ""
    sglang_model: str = ""
    sglang_api_key: str = "EMPTY"
    sglang_timeout_s: float = Field(default=120.0, gt=0)
    sglang_max_workers: int = Field(default=8, ge=1)
    sglang_sync_weights: bool = True
    sglang_weight_dir: str = "sglang_weight_sync"
    sglang_weight_mode: str = "full"
    sglang_weight_keep: int = Field(default=2, ge=1)
    sglang_engine_urls: list[str] | str | None = None
    vllm_base_url: str = ""
    vllm_model: str = ""
    vllm_api_key: str = "EMPTY"
    vllm_timeout_s: float = Field(default=120.0, gt=0)
    vllm_max_workers: int = Field(default=8, ge=1)
    vllm_sync_weights: bool = True
    vllm_weight_dir: str = "vllm_weight_sync"
    vllm_weight_mode: str = "auto"
    vllm_weight_keep: int = Field(default=2, ge=1)
    vllm_engine_urls: list[str] | str | None = None
    vllm_lora_name: str = "seiso_slime_policy"
    require_thinking_trace: bool = True
    thinking_instruction: str = Field(
        default="Show your reasoning in <think>...</think>, then give the final answer.",
        min_length=1,
    )
    outcome_reward_weight: float = Field(default=1.0, ge=0)
    format_reward_weight: float = Field(
        default=0.1,
        ge=0,
        description="Weight for closed <think>...</think> format on generated tokens only.",
    )
    process_reward_weight: float = Field(
        default=0.0,
        ge=0,
        description="Experimental lexical process score; keep 0 for verifiable outcome-first RL.",
    )
    missing_thinking_penalty: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Optional subtractive penalty when thinking format is required but "
            "missing. Prefer format_reward_weight for shaping; keep 0 by default."
        ),
    )
    code_reward_mode: str = Field(
        default="binary",
        description=(
            "Code GRPO outcome mapping: binary (all unit tests pass, default), "
            "dense (pass fraction), or auto (dense until a group has a full passer)."
        ),
    )
    slime_eval_dataset: Path | None = Field(
        default=None,
        description=(
            "Frozen held-out JSONL for slime unit-test eval (must differ from dataset)."
        ),
    )
    slime_eval_every_steps: int = Field(
        default=0,
        ge=0,
        description="Held-out eval cadence; 0 means only at end when slime_eval_on_complete.",
    )
    slime_eval_max_prompts: int | None = Field(default=None, ge=1)
    slime_eval_on_complete: bool = True
    min_thinking_tokens: int = Field(default=8, ge=0)
    dtype: str = "auto"
    device: str = "cuda"
    slime_use_lora: bool = True
    use_8bit_optimizer: bool = False
    trust_remote_code: bool = False
    best_checkpoint_dir: str = Field(default="checkpoint-best", min_length=1)
    final_checkpoint_dir: str = ""
    auto_stop: bool = True
    auto_stop_metric: str = "outcome_reward_mean"
    auto_stop_patience: int = Field(default=20, ge=1)
    auto_stop_min_delta: float = Field(default=1e-4, ge=0)
    auto_stop_warmup_steps: int = Field(default=10, ge=0)
    stop_on_nonfinite: bool = True
    write_verifier_data: bool = True
    verifier_data_file: str = Field(default="slime_verifier_data.jsonl", min_length=1)
    verifier_max_text_chars: int = Field(default=2048, ge=0)
    # Opt-in grounded corpus materialize (dataset / data_designer).
    data_gen: bool = False
    data_gen_count: int = Field(default=0, ge=0)
    data_gen_seed: int = 0
    data_gen_mix: str = "numeric:0.7,choice:0.3"
    data_gen_difficulty: str = "easy:0.35,medium:0.45,hard:0.20"
    data_gen_filename: str = "slime_generated.jsonl"
    data_gen_source: str = Field(
        default="off",
        description="Materialize source: off | dataset | data_designer | auto",
    )
    dataset_ref: str | None = Field(
        default=None,
        validation_alias=AliasChoices("dataset_ref", "hf_dataset"),
        description="HF hub id / path when data_gen_source=dataset",
    )
    dataset_split: str = "train"
    data_designer: str = Field(
        default="off",
        description='NVIDIA NeMo Data Designer: off | on | auto (quote "on"/"off" in YAML)',
    )

    @field_validator("data_designer", mode="before")
    @classmethod
    def _coerce_data_designer(cls, value: Any) -> str:
        # YAML 1.1 parses bare `on`/`off` as booleans.
        if value is True:
            return "on"
        if value is False:
            return "off"
        return str(value if value is not None else "off")

    @field_validator("data_gen_source", mode="before")
    @classmethod
    def _coerce_data_gen_source(cls, value: Any) -> str:
        if value is True:
            return "auto"
        if value is False:
            return "off"
        text = str(value if value is not None else "off")
        if text.strip().lower() == "hf_dataset":
            return "dataset"
        return text
    require_held_out_eval: bool = Field(
        default=True,
        description="Require disjoint eval_dataset for product slime runs",
    )
    vllm_tensor_parallel: int = Field(
        default=0,
        ge=0,
        description="Optional vLLM TP size hint for multi-GPU Data Designer gate",
    )
    # ── NVIDIA NeMo RL (external; requires SEISO_NEMO_RL_ROOT) ──
    nemo_rl_root: Path | None = Field(
        default=None,
        description="Path to a NVIDIA-NeMo/RL checkout (else SEISO_NEMO_RL_ROOT).",
    )
    nemo_rl_recipe: str = Field(
        default="grpo",
        description="NeMo RL recipe: grpo | dpo | distillation | smoke",
    )
    nemo_rl_base_config: str | None = Field(
        default=None,
        description="Optional NeMo RL YAML relative to nemo_rl_root (recipe default).",
    )
    nemo_rl_gpus_per_node: int = Field(default=1, ge=1)
    nemo_rl_num_nodes: int = Field(default=1, ge=1)
    nemo_rl_max_steps: int | None = Field(default=None, ge=1)
    nemo_rl_use_lora: bool = False
    nemo_rl_extra_overrides: list[str] = Field(
        default_factory=list,
        description="Extra Hydra overrides passed to NeMo RL (e.g. logger.wandb_enabled=False).",
    )
    nemo_rl_dry_run: bool = Field(
        default=False,
        description="Write launch sidecar + manifest without executing uv/NeMo RL.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Advanced overrides, including moe_finetune, freeze_moe_router, "
            "and lora_target_modules."
        ),
    )

    @field_validator(
        "output_dir",
        "dataset",
        "resume_from",
        "sandbox_root",
        "nemo_rl_root",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, v: Any) -> Any:
        if v is None:
            return v
        return Path(v).expanduser()

    @field_validator(
        "distributed_master_addr",
        "ddp_backend",
        "cloud_gpu_region",
        "cloud_gpu_instance_type",
        "cloud_gpu_project",
    )
    @classmethod
    def _validate_safe_runtime_label(cls, v: str | None) -> str | None:
        if v is None:
            return v
        lowered = v.lower()
        forbidden = ("token", "secret", "password", "apikey", "api_key", "://")
        if any(marker in lowered for marker in forbidden):
            raise ValueError("runtime labels cannot contain secrets, URLs, or tokens")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-:/ ")
        if any(ch not in allowed for ch in v):
            raise ValueError(
                "runtime labels may only use letters, numbers, spaces, '.', '_', '-', ':', '/'"
            )
        return v

    @field_validator("dynamic_sampling_filter")
    @classmethod
    def _validate_dynamic_sampling_filter(cls, v: str) -> str:
        if v not in {"none", "reward_nonzero_std", "outcome_nonzero_std"}:
            raise ValueError(
                "dynamic_sampling_filter must be one of: "
                "none, reward_nonzero_std, outcome_nonzero_std"
            )
        return v

    @model_validator(mode="after")
    def _validate_cloud_gpu_config(self) -> TrainConfig:
        if not self.cloud_gpu_enabled:
            return self
        if self.cloud_gpu_provider == CloudGpuProvider.NONE:
            raise ValueError("cloud_gpu_provider is required when cloud_gpu_enabled is true")
        if not self.cloud_gpu_instance_type.strip():
            raise ValueError("cloud_gpu_instance_type is required when cloud_gpu_enabled is true")
        # Cloud GPU fields are provisioning metadata only — they do not launch
        # vLLM. Slime + explicit vLLM rollouts need a reachable engine URL.
        # rollout_backend=auto without a URL stays on HF (valid); managed vLLM
        # may still be adopted at runtime when already running.
        backend = str(self.rollout_backend or "hf").strip().lower()
        if self.method == TrainMethod.SLIME and backend == "vllm":
            has_url = bool(str(self.vllm_base_url or "").strip())
            if not has_url:
                raise ValueError(
                    "cloud_gpu_enabled with method=slime and rollout_backend=vllm "
                    "requires vllm_base_url pointing at a running multi-GPU vLLM "
                    "server (cloud_gpu_* does not auto-start the engine; see "
                    "docs/training/multi-gpu.md)"
                )
        return self

    @model_validator(mode="after")
    def _validate_meaningful_algorithms(self) -> TrainConfig:
        # AUTO is treated like chat: packing + response-only masks are incompatible
        # until the format is known to be plain TEXT (set dataset_format=text for CPT).
        packing_blocked_formats = {
            DatasetFormat.AUTO,
            DatasetFormat.CHAT,
            DatasetFormat.SHAREGPT,
            DatasetFormat.ALPACA,
            DatasetFormat.PREFERENCE,
        }
        if (
            self.packing
            and self.train_on_responses_only
            and self.dataset_format in packing_blocked_formats
        ):
            raise ValueError(
                "packing cannot be combined with train_on_responses_only for "
                f"{self.dataset_format.value} datasets; use packing only for plain "
                "text (dataset_format=text) or disable train_on_responses_only"
            )
        if self.packing and bool(getattr(self, "assistant_only_loss", False)):
            raise ValueError(
                "packing cannot be combined with assistant_only_loss "
                "(Seiso renders packed text without TRL assistant span masks)"
            )
        if self.dataset_format == DatasetFormat.PREFERENCE:
            if self.method == TrainMethod.SLIME:
                raise ValueError(
                    "Preference datasets are incompatible with method=slime "
                    "(GRPO needs verifiable prompt/answer rows). "
                    "Use Distill-RL/DPO for preference pairs, or LoRA/full with "
                    "preference_as_sft=true for chosen-only SFT."
                )
            if self.method == TrainMethod.NEMO_RL and self.nemo_rl_recipe not in {
                "dpo",
            }:
                raise ValueError(
                    "Preference datasets with method=nemo_rl require "
                    "nemo_rl_recipe=dpo (GRPO/smoke use verifiable prompts). "
                    "Or use Distill-RL/DPO, or preference_as_sft=true for SFT."
                )
            nemo_rl_dpo = (
                self.method == TrainMethod.NEMO_RL and self.nemo_rl_recipe == "dpo"
            )
            if not nemo_rl_dpo and not self.preference_as_sft:
                raise ValueError(
                    "Preference datasets (chosen/rejected) are not SFT alignment. "
                    "Use Distill-RL/DPO (`seiso distill-rl`) or method=nemo_rl with "
                    "nemo_rl_recipe=dpo for real preference learning, "
                    "or set preference_as_sft=true to train supervised on chosen responses "
                    "only (rejected pairs are discarded)."
                )
        if self.method == TrainMethod.FULL and self.quant in (
            QuantMode.INT4,
            QuantMode.INT8,
        ):
            raise ValueError(
                f"method=full cannot use quant={self.quant.value}; use method=lora "
                "(QLoRA) or quant=none/16bit for full fine-tuning"
            )
        return self

    @model_validator(mode="after")
    def _validate_slime_batch_and_clip(self) -> TrainConfig:
        if self.method != TrainMethod.SLIME:
            return self
        from seiso.slime.config import validate_oversample_vs_train_batch

        train_batch = self.train_batch_size or self.rollout_batch_size or self.batch_size
        validate_oversample_vs_train_batch(
            dynamic_sampling_filter=self.dynamic_sampling_filter,
            over_sampling_batch_size=self.over_sampling_batch_size,
            train_batch_size=train_batch,
            rollout_batch_size=self.rollout_batch_size,
        )
        # Mirror SingleGpuSlimeConfig: multi-epoch needs a small KL trust region.
        if self.epochs > 1 and self.kl_coef == 0.0:
            import os

            allow_zero = os.environ.get("SEISO_SLIME_ALLOW_ZERO_KL", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            if not allow_zero:
                object.__setattr__(self, "kl_coef", 0.02)
        if self.clip_ratio_high is not None and self.clip_ratio_high < self.clip_ratio:
            raise ValueError("clip_ratio_high must be >= clip_ratio")
        if self.outcome_reward_weight <= 0:
            raise ValueError(
                "outcome_reward_weight must be > 0 for meaningful GRPO "
                "(verifiable outcome signal required)"
            )
        shaping = self.format_reward_weight + self.process_reward_weight
        if shaping >= self.outcome_reward_weight:
            raise ValueError(
                "format_reward_weight + process_reward_weight must be strictly "
                "less than outcome_reward_weight (outcome must dominate; ties "
                "allow format bias)"
            )
        if self.require_thinking_trace:
            headroom = self.outcome_reward_weight - (
                self.format_reward_weight + self.process_reward_weight
            )
            if self.missing_thinking_penalty >= headroom:
                raise ValueError(
                    "missing_thinking_penalty must be < outcome_reward_weight - "
                    "(format_reward_weight + process_reward_weight) so "
                    "correct-but-unformatted completions strictly outrank "
                    "wrong-but-formatted ones (ties allow format bias)"
                )
        from seiso.rl_verify.verify import resolve_code_reward_mode

        try:
            object.__setattr__(
                self,
                "code_reward_mode",
                resolve_code_reward_mode(self.code_reward_mode),
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        # Fail loud at TrainConfig validation (Forge/API start) — not after
        # the job is queued — when held-out eval / slime invariants are missing.
        try:
            self.to_single_gpu_slime_config().validate()
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_validator(mode="after")
    def _validate_nemo_rl(self) -> TrainConfig:
        if self.method != TrainMethod.NEMO_RL:
            return self
        from seiso.slime.config import allow_tiny_rl, is_slime_ci_fixture_path

        # NeMo recipes use their own corpora; Seiso still requires dataset for
        # TrainConfig. Do not advertise CI toys as the product dataset field.
        if not allow_tiny_rl() and is_slime_ci_fixture_path(self.dataset):
            raise ValueError(
                f"dataset={self.dataset} is a slime CI fixture "
                "(data/slime_*.jsonl). For NeMo RL examples use a Hub id "
                "placeholder (recipes ship their own data) or a real JSONL. "
                "Smoke/CI: SEISO_ALLOW_TINY_RL=1."
            )
        try:
            self.to_nemo_rl_config().validate()
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("training config must be a mapping")
        # Accept slime / SingleGpuSlimeConfig field names so example YAMLs
        # work with both `seiso train` and `seiso slime`.
        aliases = {
            "save_every_steps": "save_steps",
            "log_every_steps": "logging_steps",
            "use_lora": "slime_use_lora",
            "eval_dataset": "slime_eval_dataset",
            "eval_every_steps": "slime_eval_every_steps",
            "eval_max_prompts": "slime_eval_max_prompts",
            "eval_on_complete": "slime_eval_on_complete",
        }
        for src, dest in aliases.items():
            if src in data and dest not in data:
                data[dest] = data.pop(src)
            elif src in data:
                data.pop(src)
        return cls.model_validate(data)

    def to_single_gpu_slime_config(self):
        """Project a general training config into the release-grade slime runner."""
        from seiso.slime.config import SingleGpuSlimeConfig

        extra: dict[str, Any] = getattr(self, "extra", {})
        # Keep GRPO groups intact in microbatches (OpenRLHF/verl practice).
        # Do not fall back to batch_size=1 when rollouts_per_prompt > 1.
        policy_batch = self.policy_micro_batch_size
        if policy_batch is None:
            policy_batch = int(self.rollouts_per_prompt)
        # slime: train target defaults to rollout_batch_size (prompts)
        train_batch = self.train_batch_size
        return SingleGpuSlimeConfig(
            model_id=self.model_id,
            dataset=Path(self.dataset),
            output_dir=self.output_dir,
            sandbox_root=self.sandbox_root,
            eval_dataset=(
                Path(self.slime_eval_dataset)
                if self.slime_eval_dataset is not None
                else None
            ),
            eval_every_steps=self.slime_eval_every_steps,
            eval_max_prompts=self.slime_eval_max_prompts,
            eval_on_complete=self.slime_eval_on_complete,
            prompt_field=self.prompt_field,
            answer_field=self.answer_field,
            metadata_field=self.metadata_field,
            reward=self.reward,
            reward_field=self.reward_field,
            max_vram_gb=self.max_vram_gb,
            max_prompt_tokens=self.max_prompt_tokens,
            max_new_tokens=self.max_new_tokens,
            rollouts_per_prompt=self.rollouts_per_prompt,
            rollout_batch_size=self.rollout_batch_size,
            over_sampling_batch_size=self.over_sampling_batch_size,
            dynamic_sampling_filter=self.dynamic_sampling_filter,
            dynamic_sampling_min_reward_std=self.dynamic_sampling_min_reward_std,
            policy_micro_batch_size=policy_batch,
            train_batch_size=train_batch,
            balance_data=self.balance_data,
            shuffle_buffer_size=self.shuffle_buffer_size,
            max_samples_per_epoch=self.max_samples_per_epoch,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            epochs=self.epochs,
            max_steps=extra.get("max_steps"),
            kl_coef=self.kl_coef,
            clip_ratio=self.clip_ratio,
            clip_ratio_high=self.clip_ratio_high,
            clip_ratio_c=self.clip_ratio_c,
            grpo_std_normalization=self.grpo_std_normalization,
            calculate_per_token_loss=self.calculate_per_token_loss,
            loss_aggregation=self.loss_aggregation,
            temperature=self.temperature,
            top_p=self.top_p,
            rollout_backend=self.rollout_backend,
            apply_chat_template=self.apply_chat_template,
            sglang_base_url=self.sglang_base_url,
            sglang_model=self.sglang_model,
            sglang_api_key=self.sglang_api_key,
            sglang_timeout_s=self.sglang_timeout_s,
            sglang_max_workers=self.sglang_max_workers,
            sglang_sync_weights=self.sglang_sync_weights,
            sglang_weight_dir=self.sglang_weight_dir,
            sglang_weight_mode=self.sglang_weight_mode,
            sglang_weight_keep=self.sglang_weight_keep,
            sglang_engine_urls=self.sglang_engine_urls,
            vllm_base_url=self.vllm_base_url,
            vllm_model=self.vllm_model,
            vllm_api_key=self.vllm_api_key,
            vllm_timeout_s=self.vllm_timeout_s,
            vllm_max_workers=self.vllm_max_workers,
            vllm_sync_weights=self.vllm_sync_weights,
            vllm_weight_dir=self.vllm_weight_dir,
            vllm_weight_mode=self.vllm_weight_mode,
            vllm_weight_keep=self.vllm_weight_keep,
            vllm_engine_urls=self.vllm_engine_urls,
            vllm_lora_name=self.vllm_lora_name,
            require_thinking_trace=self.require_thinking_trace,
            thinking_instruction=self.thinking_instruction,
            outcome_reward_weight=self.outcome_reward_weight,
            format_reward_weight=self.format_reward_weight,
            process_reward_weight=self.process_reward_weight,
            missing_thinking_penalty=self.missing_thinking_penalty,
            code_reward_mode=self.code_reward_mode,
            min_thinking_tokens=self.min_thinking_tokens,
            seed=self.seed,
            dtype=self.dtype,
            device=self.device,
            gradient_checkpointing=self.gradient_checkpointing,
            use_lora=self.slime_use_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_target_modules=extra.get("lora_target_modules"),
            lora_bias=str(extra.get("lora_bias", "none")),
            use_8bit_optimizer=self.use_8bit_optimizer,
            trust_remote_code=self.trust_remote_code,
            save_every_steps=self.save_steps,
            log_every_steps=self.logging_steps,
            best_checkpoint_dir=self.best_checkpoint_dir,
            final_checkpoint_dir=self.final_checkpoint_dir,
            auto_stop=self.auto_stop,
            auto_stop_metric=self.auto_stop_metric,
            auto_stop_patience=self.auto_stop_patience,
            auto_stop_min_delta=self.auto_stop_min_delta,
            auto_stop_warmup_steps=self.auto_stop_warmup_steps,
            stop_on_nonfinite=self.stop_on_nonfinite,
            write_verifier_data=self.write_verifier_data,
            verifier_data_file=self.verifier_data_file,
            verifier_max_text_chars=self.verifier_max_text_chars,
            data_gen=self.data_gen,
            data_gen_count=self.data_gen_count,
            data_gen_seed=self.data_gen_seed,
            data_gen_mix=self.data_gen_mix,
            data_gen_difficulty=self.data_gen_difficulty,
            data_gen_filename=self.data_gen_filename,
            data_gen_source=self.data_gen_source,
            dataset_ref=self.dataset_ref,
            dataset_split=self.dataset_split,
            data_designer=self.data_designer,
            require_held_out_eval=self.require_held_out_eval,
            vllm_tensor_parallel=self.vllm_tensor_parallel,
        )

    def to_nemo_rl_config(self):
        """Project training config into the NeMo RL external launcher."""
        from seiso.nemo_rl.config import NeMoRLConfig

        return NeMoRLConfig.from_mapping(
            {
                "model_id": self.model_id,
                "output_dir": self.output_dir,
                "recipe": self.nemo_rl_recipe,
                "nemo_rl_root": self.nemo_rl_root,
                "base_config": self.nemo_rl_base_config,
                "gpus_per_node": self.nemo_rl_gpus_per_node,
                "num_nodes": self.nemo_rl_num_nodes,
                "max_steps": self.nemo_rl_max_steps,
                "learning_rate": self.learning_rate,
                "rollouts_per_prompt": self.rollouts_per_prompt,
                "num_prompts_per_step": self.rollout_batch_size,
                "seed": self.seed,
                "use_lora": self.nemo_rl_use_lora,
                "extra_overrides": tuple(self.nemo_rl_extra_overrides or ()),
                "dry_run": self.nemo_rl_dry_run,
                "sandbox_root": self.sandbox_root,
                "extra": dict(self.extra or {}),
            }
        )


def run_training(
    config: TrainConfig,
    *,
    on_metric=None,
    on_log: Callable[[str], None] | None = None,
    job_id: str | None = None,
) -> Path:
    """Execute training job; returns output checkpoint directory."""
    from seiso.env import configure_transformers_env
    from seiso.models.hf_env import configure_hf_hub_cache
    from seiso.platform import ensure_cuda_library_path
    from seiso.security.nvidia_boundary import enforce_nvidia_secure_boundary
    from seiso.training.trainer import SeisoTrainer

    configure_transformers_env()
    if not os.environ.get("HF_HOME"):
        configure_hf_hub_cache(config.sandbox_root)
    ensure_cuda_library_path()
    enforce_nvidia_secure_boundary(context="training")
    from seiso.training.torch_dynamo import configure_compile_checkpoint_compat

    configure_compile_checkpoint_compat(
        torch_compile=config.torch_compile,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    if config.method == TrainMethod.SLIME:
        from seiso.slime.trainer import train_slime
        from seiso.training.metrics import is_main_process

        slime_config = config.to_single_gpu_slime_config()
        out = train_slime(slime_config)
        if is_main_process():
            _write_slime_manifest(config, out)
        return out
    if config.method == TrainMethod.NEMO_RL:
        from seiso.nemo_rl.runner import train_nemo_rl
        from seiso.training.metrics import is_main_process

        if on_log:
            on_log(
                "NeMo RL: launching external NVIDIA-NeMo/RL via uv "
                "(set SEISO_NEMO_RL_ROOT if the checkout is not auto-discovered)."
            )
        out = train_nemo_rl(config.to_nemo_rl_config())
        if is_main_process():
            # Manifest is written by the NeMo RL runner; ensure path exists.
            manifest = out / "seiso_manifest.json"
            if not manifest.is_file():
                _write_nemo_rl_manifest(config, out)
        return out
    trainer = SeisoTrainer(config, on_metric=on_metric, on_log=on_log, job_id=job_id)
    return trainer.run()


def _write_slime_manifest(config: TrainConfig, output_dir: Path) -> None:
    distributed = bool(
        config.multi_gpu
        or config.distributed_strategy == DistributedStrategy.DDP
        or config.distributed_num_nodes > 1
    )
    payload = {
        "model_id": config.model_id,
        "original_model_id": str(config.extra.get("original_model_id") or config.model_id),
        "method": TrainMethod.SLIME.value,
        "methodology": config.training_methodology,
        "post_training_algorithm": (
            "distributed_slime_grpo" if distributed else "single_gpu_slime_grpo"
        ),
        "adapter": "lora" if config.slime_use_lora else "full",
        "quant": config.quant.value,
        "dataset": str(config.dataset),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_vram_gb": config.max_vram_gb,
        "reward": config.reward,
        "metadata_field": config.metadata_field,
        "require_thinking_trace": config.require_thinking_trace,
        "outcome_reward_weight": config.outcome_reward_weight,
        "format_reward_weight": config.format_reward_weight,
        "process_reward_weight": config.process_reward_weight,
        "missing_thinking_penalty": config.missing_thinking_penalty,
        "code_reward_mode": config.code_reward_mode,
        "slime_eval_dataset": (
            str(config.slime_eval_dataset) if config.slime_eval_dataset else None
        ),
        "slime_eval_every_steps": config.slime_eval_every_steps,
        "slime_eval_max_prompts": config.slime_eval_max_prompts,
        "slime_eval_on_complete": config.slime_eval_on_complete,
        "min_thinking_tokens": config.min_thinking_tokens,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "over_sampling_batch_size": config.over_sampling_batch_size,
        "dynamic_sampling_filter": config.dynamic_sampling_filter,
        "clip_ratio": config.clip_ratio,
        "clip_ratio_high": config.clip_ratio_high,
        "grpo_std_normalization": config.grpo_std_normalization,
        "calculate_per_token_loss": config.calculate_per_token_loss,
        "balance_data": config.balance_data,
        "auto_stop": config.auto_stop,
        "auto_stop_metric": config.auto_stop_metric,
        "auto_stop_patience": config.auto_stop_patience,
        "best_checkpoint_dir": str(config.output_dir / config.best_checkpoint_dir),
        "verifier_data_file": (
            str(config.output_dir / config.verifier_data_file)
            if config.write_verifier_data
            else None
        ),
        "distributed": distributed,
        "distributed_strategy": config.distributed_strategy.value,
        "distributed_nproc_per_node": config.distributed_nproc_per_node,
        "distributed_num_nodes": config.distributed_num_nodes,
        "distributed_node_rank": config.distributed_node_rank,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seiso_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_nemo_rl_manifest(config: TrainConfig, output_dir: Path) -> None:
    payload = {
        "model_id": config.model_id,
        "original_model_id": str(config.extra.get("original_model_id") or config.model_id),
        "method": TrainMethod.NEMO_RL.value,
        "methodology": config.training_methodology,
        "framework": "nemo_rl",
        "upstream": "https://github.com/NVIDIA-NeMo/RL",
        "post_training_algorithm": f"nemo_rl_{config.nemo_rl_recipe}",
        "recipe": config.nemo_rl_recipe,
        "adapter": "lora" if config.nemo_rl_use_lora else "full",
        "quant": config.quant.value,
        "dataset": str(config.dataset),
        "learning_rate": config.learning_rate,
        "gpus_per_node": config.nemo_rl_gpus_per_node,
        "num_nodes": config.nemo_rl_num_nodes,
        "max_steps": config.nemo_rl_max_steps,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seiso_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
