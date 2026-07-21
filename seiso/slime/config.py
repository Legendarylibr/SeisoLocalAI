"""Configuration for slime-style RL training (single-GPU, multi-GPU, remote rollouts)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def allow_tiny_rl() -> bool:
    return os.environ.get("SEISO_ALLOW_TINY_RL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def effective_train_batch_size(config: SingleGpuSlimeConfig) -> int:
    """Prompt groups per step after filtering (slime rollout_batch_size target)."""
    if config.train_batch_size is not None:
        return int(config.train_batch_size)
    return int(config.rollout_batch_size)


def validate_oversample_vs_train_batch(
    *,
    dynamic_sampling_filter: str,
    over_sampling_batch_size: int | None,
    train_batch_size: int,
    rollout_batch_size: int | None = None,
) -> None:
    """Shared oversample rule (slime: over_sampling_batch_size >= rollout_batch_size).

    Units are **prompts**, not sequences. ``None`` oversample = no extra headroom.
    """
    if dynamic_sampling_filter == "none":
        return
    if over_sampling_batch_size is None:
        return
    if over_sampling_batch_size < 1:
        raise ValueError("over_sampling_batch_size must be positive")
    # Prefer comparing to rollout_batch_size (slime); fall back to train target.
    floor = int(rollout_batch_size if rollout_batch_size is not None else train_batch_size)
    if over_sampling_batch_size < floor:
        raise ValueError(
            "over_sampling_batch_size must be >= rollout_batch_size when "
            "dynamic_sampling_filter is enabled (slime oversample ≥ rollout batch; "
            "units are prompts)"
        )


def _normalize_data_gen_source(value: str) -> str:
    key = str(value or "off").lower().strip()
    if key == "hf_dataset":
        return "dataset"
    return key


@dataclass(frozen=True)
class SingleGpuSlimeConfig:
    """Small, explicit config for local GRPO-style training.

    Alias: ``SlimeConfig``. Supports single-GPU HF, multi-GPU DDP, and remote
    SGLang/vLLM rollouts via ``rollout_backend``.
    """

    model_id: str
    dataset: Path
    output_dir: Path
    # Optional user-data root; local dataset_ref paths must stay inside it.
    sandbox_root: Path | None = None
    # Frozen held-out prompts (not used for GRPO rollouts). Prefer a disjoint
    # unit-test JSONL such as data/slime_code_eval.jsonl.
    eval_dataset: Path | None = None
    eval_every_steps: int = 0  # 0 = only at end when eval_on_complete
    eval_max_prompts: int | None = None
    eval_on_complete: bool = True
    # slime: --input-key / --label-key / --metadata-key
    prompt_field: str = "prompt"
    answer_field: str = "label"
    metadata_field: str | None = "metadata"
    # slime: --rm-type (or per-row metadata.rm_type with reward=auto)
    reward: str = "auto"
    reward_field: str = "reward"
    max_vram_gb: float | None = None
    max_prompt_tokens: int = 512
    max_new_tokens: int = 256
    # slime: --n-samples-per-prompt
    rollouts_per_prompt: int = 4
    # slime: --rollout-batch-size (number of *prompts*, not sequences)
    rollout_batch_size: int = 1
    # slime: --over-sampling-batch-size (>= rollout_batch_size when filtering)
    over_sampling_batch_size: int | None = None
    # Drop zero-signal groups (slime check_reward_nonzero_std on *outcome*).
    dynamic_sampling_filter: str = "reward_nonzero_std"
    dynamic_sampling_min_reward_std: float = 1e-6
    policy_micro_batch_size: int = 4
    # Target prompt groups per policy step after filtering (defaults to rollout_batch_size).
    train_batch_size: int | None = None
    balance_data: bool = False
    shuffle_buffer_size: int = 2048
    max_samples_per_epoch: int | None = None
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    epochs: int = 1
    max_steps: int | None = None
    # 0 = no frozen ref (lower VRAM). Use ~0.01–0.05 for longer runs to limit drift.
    kl_coef: float = 0.0
    # slime: --eps-clip / --eps-clip-high
    clip_ratio: float = 0.2
    clip_ratio_high: float | None = None
    # slime: grpo_std_normalization (mean-center then / unbiased std)
    grpo_std_normalization: bool = True
    # Per-token importance ratios are length-stable vs exp(ΣΔlogπ) on full sequences.
    calculate_per_token_loss: bool = True
    temperature: float = 0.9
    top_p: float = 0.95
    # Online generate: hf (colocated, default) | sglang | vllm | auto
    rollout_backend: str = "hf"
    # slime: --apply-chat-template
    apply_chat_template: bool = True
    sglang_base_url: str = ""
    sglang_model: str = ""
    sglang_api_key: str = "EMPTY"
    sglang_timeout_s: float = 120.0
    sglang_max_workers: int = 8
    # After each optimizer step, rank0 writes HF weights and hot-reloads SGLang
    # (slime disk transport). Keep true for on-policy multi-GPU rollouts.
    sglang_sync_weights: bool = True
    sglang_weight_dir: str = "sglang_weight_sync"
    # full = always complete HF ckpt; delta = skip if unchanged, try /pull_weights then full
    sglang_weight_mode: str = "full"
    # Keep last N weight_v* directories under sglang_weight_dir
    sglang_weight_keep: int = 2
    # Extra engines (comma list or YAML list); sglang_base_url may also be comma-separated
    sglang_engine_urls: list[str] | str | None = None
    # Multi-GPU rollouts via OpenAI-compatible vLLM (managed multi-GPU or external).
    vllm_base_url: str = ""
    vllm_model: str = ""
    vllm_api_key: str = "EMPTY"
    vllm_timeout_s: float = 120.0
    vllm_max_workers: int = 8
    vllm_sync_weights: bool = True
    vllm_weight_dir: str = "vllm_weight_sync"
    # auto = LoRA when use_lora/PEFT else full; lora = /v1/load_lora_adapter; full = disk reload
    vllm_weight_mode: str = "auto"
    vllm_weight_keep: int = 2
    vllm_engine_urls: list[str] | str | None = None
    vllm_lora_name: str = "seiso_slime_policy"
    require_thinking_trace: bool = True
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    )
    outcome_reward_weight: float = 1.0
    # Format is a small soft bonus for closed <think>...</think> on raw tokens
    # (scaled by min_thinking_tokens; empty traces earn ~0). Prefer this over
    # missing_thinking_penalty so correct answers are not punished.
    format_reward_weight: float = 0.1
    # Lexical process shaping is experimental; leave at 0 for verifiable outcome-first RL.
    process_reward_weight: float = 0.0
    # Optional subtractive push for missing think tags. Default 0 — use the format
    # bonus alone; set a modest value (e.g. 0.2) only if format compliance stalls.
    missing_thinking_penalty: float = 0.0
    # Code GRPO outcome: binary (all tests pass, default) | dense (pass fraction)
    # | auto (dense until a same-prompt group has a full passer, then binary).
    code_reward_mode: str = "binary"
    min_thinking_tokens: int = 8
    seed: int = 17
    dtype: str = "auto"
    device: str = "cuda"
    gradient_checkpointing: bool = True
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    lora_bias: str = "none"
    use_8bit_optimizer: bool = False
    trust_remote_code: bool = False
    save_every_steps: int = 100
    log_every_steps: int = 1
    best_checkpoint_dir: str = "checkpoint-best"
    final_checkpoint_dir: str = ""
    auto_stop: bool = True
    auto_stop_metric: str = "reward_mean"
    auto_stop_patience: int = 20
    auto_stop_min_delta: float = 1e-4
    auto_stop_warmup_steps: int = 10
    stop_on_nonfinite: bool = True
    write_verifier_data: bool = True
    verifier_data_file: str = "slime_verifier_data.jsonl"
    verifier_max_text_chars: int = 2048
    # High-level data generation: when enabled (or count > 0), materialize a
    # verifiable prompt corpus before training. Completions still come from
    # online rollouts. Prefer operator/HF ``dataset`` without data_gen.
    data_gen: bool = False
    data_gen_count: int = 0
    data_gen_seed: int = 0
    data_gen_mix: str = "numeric:0.7,choice:0.3"
    data_gen_difficulty: str = "easy:0.35,medium:0.45,hard:0.20"
    data_gen_filename: str = "slime_generated.jsonl"
    # Materialize source: off (default) | dataset | data_designer | auto.
    data_gen_source: str = "off"
    # HF hub id / path when data_gen is on and data_gen_source=dataset|auto.
    dataset_ref: str | None = None
    dataset_split: str = "train"
    # NVIDIA NeMo Data Designer: on = force; off = never; auto = no silent select
    # (pair with data_gen_source=data_designer for opt-in materialize).
    data_designer: str = "off"
    # Optional TP hint for gate when WORLD_SIZE==1 but vLLM uses multiple GPUs.
    vllm_tensor_parallel: int = 0
    # Product default: require a disjoint held-out eval JSONL.
    # data_gen may auto-split as a last resort (warning). CI: SEISO_ALLOW_TINY_RL=1.
    require_held_out_eval: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> SingleGpuSlimeConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("slime config must be a mapping")
        # Accept TrainConfig field names so shared example YAMLs work with
        # both `seiso slime` and `seiso train -c ... method: slime`.
        aliases = {
            "save_steps": "save_every_steps",
            "logging_steps": "log_every_steps",
            "slime_use_lora": "use_lora",
            "hf_dataset": "dataset_ref",
        }
        for src, dest in aliases.items():
            if src in data and dest not in data:
                data[dest] = data.pop(src)
            elif src in data:
                data.pop(src)
        known = {f.name for f in fields(cls)}
        # Filter unknown keys (e.g. method/quant from TrainConfig-oriented YAMLs).
        path_keys = {"dataset", "output_dir", "eval_dataset"}
        payload: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            if key in path_keys and value is not None:
                payload[key] = Path(value)
            elif key in {"data_designer", "data_gen_source"} and isinstance(value, bool):
                # YAML 1.1 parses bare on/off as booleans.
                if key == "data_designer":
                    payload[key] = "on" if value else "off"
                else:
                    payload[key] = "auto" if value else "off"
            else:
                payload[key] = value
        if "data_gen_source" in payload:
            payload["data_gen_source"] = _normalize_data_gen_source(
                str(payload["data_gen_source"])
            )
        return cls(**payload)

    def validate(self) -> None:
        if self.rollouts_per_prompt < 2:
            raise ValueError("rollouts_per_prompt must be at least 2 for grouped advantages")
        if self.train_batch_size is not None and self.train_batch_size < 1:
            raise ValueError("train_batch_size must be positive")
        if self.metadata_field is not None and not self.metadata_field:
            raise ValueError("metadata_field must not be empty")
        if self.rollout_batch_size < 1:
            raise ValueError("rollout_batch_size must be positive (slime: prompt count)")
        if self.dynamic_sampling_filter not in {
            "none",
            "reward_nonzero_std",
            "outcome_nonzero_std",
        }:
            raise ValueError(
                "dynamic_sampling_filter must be one of: "
                "none, reward_nonzero_std, outcome_nonzero_std"
            )
        validate_oversample_vs_train_batch(
            dynamic_sampling_filter=self.dynamic_sampling_filter,
            over_sampling_batch_size=self.over_sampling_batch_size,
            train_batch_size=effective_train_batch_size(self),
            rollout_batch_size=self.rollout_batch_size,
        )
        if self.dynamic_sampling_min_reward_std < 0:
            raise ValueError("dynamic_sampling_min_reward_std must be non-negative")
        if self.policy_micro_batch_size < 1:
            raise ValueError("policy_micro_batch_size must be positive")
        if self.shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")
        if self.max_samples_per_epoch is not None and self.max_samples_per_epoch < 1:
            raise ValueError("max_samples_per_epoch must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.max_prompt_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("token limits must be positive")
        if self.kl_coef < 0:
            raise ValueError("kl_coef must be non-negative")
        # Multi-epoch online GRPO without a trust region drifts; apply a small
        # default KL unless the operator opts into zero-KL (VRAM) via env.
        if self.epochs > 1 and self.kl_coef == 0.0:
            import os

            allow_zero = os.environ.get("SEISO_SLIME_ALLOW_ZERO_KL", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            if not allow_zero:
                import logging

                logging.getLogger(__name__).info(
                    "epochs=%s with kl_coef=0: applying kl_coef=0.02 for multi-epoch "
                    "trust region (set SEISO_SLIME_ALLOW_ZERO_KL=1 to keep kl_coef=0)",
                    self.epochs,
                )
                object.__setattr__(self, "kl_coef", 0.02)
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive")
        if self.clip_ratio_high is not None and self.clip_ratio_high < self.clip_ratio:
            raise ValueError(
                "clip_ratio_high must be >= clip_ratio (slime eps_clip_high >= eps_clip)"
            )
        if not self.thinking_instruction:
            raise ValueError("thinking_instruction must not be empty")
        if self.outcome_reward_weight <= 0:
            raise ValueError(
                "outcome_reward_weight must be > 0 for meaningful GRPO "
                "(verifiable outcome signal required)"
            )
        if self.format_reward_weight < 0:
            raise ValueError("format_reward_weight must be non-negative")
        if self.process_reward_weight < 0:
            raise ValueError("process_reward_weight must be non-negative")
        shaping = self.format_reward_weight + self.process_reward_weight
        if shaping > self.outcome_reward_weight:
            raise ValueError(
                "format_reward_weight + process_reward_weight must not exceed "
                "outcome_reward_weight (outcome must dominate for meaningful GRPO)"
            )
        if self.missing_thinking_penalty < 0:
            raise ValueError("missing_thinking_penalty must be non-negative")
        from seiso.rl_verify.verify import resolve_code_reward_mode

        try:
            object.__setattr__(
                self,
                "code_reward_mode",
                resolve_code_reward_mode(self.code_reward_mode),
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        # Correct-unformatted reward is outcome - penalty; wrong-formatted is at
        # most format + process. Require outcome - penalty >= format + process
        # (equivalently penalty <= outcome - shaping) so ranking cannot invert.
        if self.require_thinking_trace:
            headroom = self.outcome_reward_weight - (
                self.format_reward_weight + self.process_reward_weight
            )
            if self.missing_thinking_penalty > headroom:
                raise ValueError(
                    "missing_thinking_penalty must be <= outcome_reward_weight - "
                    "(format_reward_weight + process_reward_weight) so "
                    "correct-but-unformatted completions are not outranked by "
                    "wrong-but-formatted ones"
                )
        if self.min_thinking_tokens < 0:
            raise ValueError("min_thinking_tokens must be non-negative")
        if self.max_vram_gb is not None and self.max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be positive")
        if self.save_every_steps < 0:
            raise ValueError("save_every_steps must be non-negative")
        if self.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")
        if self.eval_every_steps < 0:
            raise ValueError("eval_every_steps must be non-negative")
        if self.eval_max_prompts is not None and self.eval_max_prompts < 1:
            raise ValueError("eval_max_prompts must be positive when set")
        if self.eval_dataset is not None and self.eval_dataset == self.dataset:
            raise ValueError(
                "eval_dataset must differ from dataset (held-out eval cannot "
                "reuse the training JSONL)"
            )
        # Only paths that can actually materialize may auto-split held-out.
        # Bare dataset_ref, source=off, or auto without ref/DD=on do not.
        src_for_split = _normalize_data_gen_source(str(self.data_gen_source or "off"))
        materialize_enabled = bool(self.data_gen or self.data_gen_count > 0)
        ref = (self.dataset_ref or "").strip()
        dd_mode = str(self.data_designer or "off").lower().strip()
        dd_on = dd_mode in {"on", "true", "1", "yes", "force", "always"}
        dd_off = dd_mode in {"off", "false", "0", "no", "disable", "disabled"}
        dataset_as_hub = bool(
            self.dataset and not Path(self.dataset).expanduser().is_file()
        )
        if not materialize_enabled or src_for_split in {"off", "none"}:
            materialize_will_split = False
        elif src_for_split == "dataset":
            materialize_will_split = bool(ref) or dataset_as_hub
        elif src_for_split == "data_designer":
            # Trainer proceeds when mode is not off (on|auto).
            materialize_will_split = not dd_off
        elif src_for_split in {"auto", ""}:
            # Matches trainer want_dataset / want_dd selection.
            materialize_will_split = bool(ref) or dd_on
        else:
            materialize_will_split = False
        if (
            self.require_held_out_eval
            and self.eval_dataset is None
            and not allow_tiny_rl()
            and not materialize_will_split
        ):
            raise ValueError(
                "eval_dataset is required for product slime runs (held-out "
                "verifiable eval, distinct from dataset). CI fixtures may set "
                "require_held_out_eval=false or SEISO_ALLOW_TINY_RL=1."
            )
        if self.use_lora:
            if self.lora_r < 1:
                raise ValueError("lora_r must be positive")
            if self.lora_alpha < 1:
                raise ValueError("lora_alpha must be positive")
            if self.lora_dropout < 0 or self.lora_dropout >= 1:
                raise ValueError("lora_dropout must be in [0, 1)")
            if self.lora_bias not in {"none", "all", "lora_only"}:
                raise ValueError("lora_bias must be one of: none, all, lora_only")
        if self.lora_target_modules is not None and not self.lora_target_modules:
            raise ValueError("lora_target_modules must not be empty")
        if self.auto_stop_patience < 1:
            raise ValueError("auto_stop_patience must be positive")
        if self.auto_stop_min_delta < 0:
            raise ValueError("auto_stop_min_delta must be non-negative")
        if self.auto_stop_warmup_steps < 0:
            raise ValueError("auto_stop_warmup_steps must be non-negative")
        if self.verifier_max_text_chars < 0:
            raise ValueError("verifier_max_text_chars must be non-negative")
        if not self.best_checkpoint_dir:
            raise ValueError("best_checkpoint_dir must not be empty")
        if self.write_verifier_data and not self.verifier_data_file:
            raise ValueError("verifier_data_file must not be empty")
        if self.data_gen_count < 0:
            raise ValueError("data_gen_count must be non-negative")
        if self.data_gen and self.data_gen_count < 1:
            raise ValueError(
                "data_gen requires data_gen_count >= 1 "
                "(prefer 200+ for meaningful GRPO outcome diversity)"
            )
        if not self.data_gen_filename:
            raise ValueError("data_gen_filename must not be empty")
        if not self.data_gen_mix:
            raise ValueError("data_gen_mix must not be empty")
        if not self.data_gen_difficulty:
            raise ValueError("data_gen_difficulty must not be empty")
        src = _normalize_data_gen_source(str(self.data_gen_source or "off"))
        if src not in {
            "auto",
            "data_designer",
            "dataset",
            "off",
            "none",
            "",
        }:
            raise ValueError(
                "data_gen_source must be one of: off, auto, data_designer, "
                f"dataset (got {self.data_gen_source!r})"
            )
        if src == "dataset" and (self.data_gen or self.data_gen_count > 0):
            ref = (self.dataset_ref or "").strip()
            dataset_as_hub = bool(
                self.dataset and not Path(self.dataset).expanduser().is_file()
            )
            if not ref and not dataset_as_hub:
                raise ValueError(
                    "data_gen_source=dataset requires dataset_ref (HF hub id) "
                    "or a non-file dataset ref"
                )
        mode = str(self.data_designer or "auto").lower().strip()
        if mode not in {
            "auto",
            "on",
            "off",
            "true",
            "false",
            "1",
            "0",
            "yes",
            "no",
            "force",
            "always",
            "disable",
            "disabled",
        }:
            raise ValueError(
                f"data_designer must be one of: auto, on, off (got {self.data_designer!r})"
            )
        if int(self.vllm_tensor_parallel or 0) < 0:
            raise ValueError("vllm_tensor_parallel must be non-negative")
        from seiso.slime.rollout_backend import (
            validate_rollout_backend_config,
        )

        validate_rollout_backend_config(self)
        mode = str(self.sglang_weight_mode or "full").lower()
        if mode not in {"full", "delta"}:
            raise ValueError("sglang_weight_mode must be 'full' or 'delta'")
        if self.sglang_weight_keep < 1:
            raise ValueError("sglang_weight_keep must be >= 1")
        vmode = str(self.vllm_weight_mode or "auto").lower()
        if vmode not in {"auto", "lora", "full"}:
            raise ValueError("vllm_weight_mode must be one of: auto, lora, full")
        if self.vllm_weight_keep < 1:
            raise ValueError("vllm_weight_keep must be >= 1")
        if not str(self.vllm_lora_name or "").strip():
            raise ValueError("vllm_lora_name must not be empty")


# Preferred name (package supports single- and multi-GPU / remote rollouts).
SlimeConfig = SingleGpuSlimeConfig
