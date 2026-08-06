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


def allow_template_slime() -> bool:
    """Shape-only YAMLs may omit real JSONL until the operator replaces paths."""
    return os.environ.get("SEISO_ALLOW_TEMPLATE_SLIME", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


# Committed toys under data/ — CI / smoke only (SEISO_ALLOW_TINY_RL=1).
_SLIME_CI_FIXTURE_NAMES = frozenset(
    {
        "slime_sample.jsonl",
        "slime_numeric_eval.jsonl",
        "slime_code_sample.jsonl",
        "slime_code_eval.jsonl",
        "slime_choice_sample.jsonl",
        "slime_choice_eval.jsonl",
    }
)

# Product grounded floor — matches Distill-RL / materialize (synth_materialize).
PRODUCT_DATA_GEN_FLOOR = 256


def min_data_gen_count_for_held_out_split(
    train_floor: int = PRODUCT_DATA_GEN_FLOOR,
) -> int:
    """Minimum ``data_gen_count`` so a 10% held-out cut still leaves ``train_floor``.

    Mirrors ``seiso.slime.trainer`` auto-split:
    ``n_eval = max(1, n // 10)``, train = ``n - n_eval``.
    """
    need = int(train_floor)
    while need - max(1, need // 10) < train_floor:
        need += 1
    return need


def is_slime_ci_fixture_path(path: Path | str | None) -> bool:
    """True when ``path`` names a committed slime CI fixture JSONL."""
    if path is None:
        return False
    try:
        name = Path(str(path)).name
    except (TypeError, ValueError):
        return False
    return name in _SLIME_CI_FIXTURE_NAMES


def looks_like_local_jsonl(path: Path | str | None) -> bool:
    """True for paths that look like local JSON/JSONL corpora (not Hub ids)."""
    if path is None:
        return False
    try:
        name = Path(str(path)).name
    except (TypeError, ValueError):
        return False
    lower = name.lower()
    return lower.endswith(".jsonl") or lower.endswith(".json")


def unresolved_local_jsonl(path: Path | str | None) -> bool:
    """True when path looks local but the file is missing on disk."""
    if not looks_like_local_jsonl(path):
        return False
    try:
        return not Path(str(path)).expanduser().is_file()
    except (TypeError, ValueError, OSError):
        return True


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


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
    # OpenRLHF / verl / slime dual-clip for negative advantages (None disables).
    clip_ratio_c: float | None = 3.0
    # slime: grpo_std_normalization (mean-center then / unbiased std)
    grpo_std_normalization: bool = True
    # Per-token importance ratios are length-stable vs exp(ΣΔlogπ) on full sequences.
    calculate_per_token_loss: bool = True
    # DeepSeekMath seq-mean (default) vs global token-mean (length-biased).
    loss_aggregation: str = "seq_mean"
    temperature: float = 0.9
    top_p: float = 0.95
    # Online generate: hf (colocated/on-policy, default) | sglang | vllm | auto.
    # For sglang/vllm, completions are sampled remotely; old_logprobs are then
    # recomputed on the local actor (engine token logprobs are not consumed), so
    # the GRPO importance ratio is slightly off-policy unless weight sync keeps
    # the engines aligned with the actor.
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
    # Prefer outcome over composite reward_mean (format shaping can inflate the latter).
    auto_stop_metric: str = "outcome_reward_mean"
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
            payload["data_gen_source"] = _normalize_data_gen_source(str(payload["data_gen_source"]))
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
        if self.dynamic_sampling_filter == "none" and not (
            allow_tiny_rl() or _env_flag("SEISO_SLIME_ALLOW_ZERO_SPREAD_GROUPS")
        ):
            raise ValueError(
                "dynamic_sampling_filter=none keeps zero-spread groups and biases "
                "GRPO toward vacuous updates; use reward_nonzero_std / "
                "outcome_nonzero_std, or set SEISO_SLIME_ALLOW_ZERO_SPREAD_GROUPS=1 "
                "(CI/smoke: SEISO_ALLOW_TINY_RL=1)"
            )
        validate_oversample_vs_train_batch(
            dynamic_sampling_filter=self.dynamic_sampling_filter,
            over_sampling_batch_size=self.over_sampling_batch_size,
            train_batch_size=effective_train_batch_size(self),
            rollout_batch_size=self.rollout_batch_size,
        )
        reward_key = str(self.reward or "auto").strip().lower()
        if reward_key == "field" and not _env_flag("SEISO_SLIME_ALLOW_FIELD_REWARD"):
            raise ValueError(
                "reward=field scores a dataset column and ignores the completion "
                "(reward hacking). Prefer reward=auto / numeric / contains_answer / "
                "code, or set SEISO_SLIME_ALLOW_FIELD_REWARD=1 only for debugging"
            )
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0 so GRPO groups have sampling diversity")
        if self.temperature > 2.0:
            raise ValueError("temperature must be <= 2.0 (degenerate sampling)")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
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
        if self.clip_ratio <= 0 or self.clip_ratio >= 1.0:
            raise ValueError(
                "clip_ratio must be in (0, 1) so the PPO/GRPO trust region lower "
                "bound stays positive (1 - clip_ratio > 0)"
            )
        if self.clip_ratio_high is not None:
            if self.clip_ratio_high < self.clip_ratio:
                raise ValueError(
                    "clip_ratio_high must be >= clip_ratio (slime eps_clip_high >= eps_clip)"
                )
            if self.clip_ratio_high > 2.0:
                raise ValueError(
                    "clip_ratio_high must be <= 2.0 (unbounded high clip disables the trust region)"
                )
        if self.clip_ratio_c is not None and float(self.clip_ratio_c) <= 1.0:
            raise ValueError(
                "clip_ratio_c must be > 1.0 for dual-clip (OpenRLHF/verl), or None to disable"
            )
        agg = str(self.loss_aggregation or "seq_mean").strip().lower()
        if agg not in {"seq_mean", "token_mean"}:
            raise ValueError("loss_aggregation must be 'seq_mean' or 'token_mean'")
        object.__setattr__(self, "loss_aggregation", agg)
        if self.policy_micro_batch_size % self.rollouts_per_prompt != 0 and not allow_tiny_rl():
            raise ValueError(
                "policy_micro_batch_size must be a multiple of rollouts_per_prompt "
                "so microbatches keep intact GRPO groups (CI: SEISO_ALLOW_TINY_RL=1)"
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
        if self.process_reward_weight > 0 and not _env_flag("SEISO_SLIME_ALLOW_PROCESS_REWARD"):
            raise ValueError(
                "process_reward_weight > 0 enables experimental lexical process "
                "shaping (gameable); leave at 0 for outcome-first GRPO, or set "
                "SEISO_SLIME_ALLOW_PROCESS_REWARD=1 to opt in"
            )
        shaping = self.format_reward_weight + self.process_reward_weight
        # Strict dominance: equality lets wrong+formatted tie correct+bare and
        # dilutes the verifiable outcome signal (reward hacking / format bias).
        if shaping >= self.outcome_reward_weight:
            raise ValueError(
                "format_reward_weight + process_reward_weight must be strictly "
                "less than outcome_reward_weight (outcome must dominate; ties "
                "allow format bias)"
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
        # Correct-unformatted = outcome - penalty; wrong-formatted ≤ format + process.
        # Require a strict gap so ranking cannot invert or tie on format alone.
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
        if self.min_thinking_tokens < 0:
            raise ValueError("min_thinking_tokens must be non-negative")
        # Composite reward_mean can plateau on format inflation; prefer outcome.
        if (
            self.auto_stop
            and str(self.auto_stop_metric or "").strip() == "reward_mean"
            and shaping > 0
            and not _env_flag("SEISO_SLIME_ALLOW_COMPOSITE_AUTOSTOP")
        ):
            import logging

            logging.getLogger(__name__).info(
                "auto_stop_metric=reward_mean with format/process shaping: "
                "using outcome_reward_mean to avoid format-biased early stop "
                "(set SEISO_SLIME_ALLOW_COMPOSITE_AUTOSTOP=1 to keep reward_mean)"
            )
            object.__setattr__(self, "auto_stop_metric", "outcome_reward_mean")
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
        dataset_as_hub = bool(self.dataset and not Path(self.dataset).expanduser().is_file())
        if not materialize_enabled or src_for_split in {"off", "none"}:
            materialize_will_split = False
        elif src_for_split == "dataset":
            materialize_will_split = bool(ref) or dataset_as_hub
        elif src_for_split == "data_designer":
            # Only credit auto-split when DD is explicitly on (auto still needs
            # package + endpoint at runtime — do not skip held-out gate).
            materialize_will_split = dd_on
        elif src_for_split in {"auto", ""}:
            # Matches trainer want_dataset / want_dd selection (dd requires on).
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
        # Product runs must not train/eval on committed CI toys. Smoke may use
        # them with SEISO_ALLOW_TINY_RL=1. data_gen may still list a fixture as
        # a temporary dataset placeholder only when materialize will replace it.
        if not allow_tiny_rl():
            if is_slime_ci_fixture_path(self.eval_dataset):
                raise ValueError(
                    f"eval_dataset={self.eval_dataset} is a slime CI fixture "
                    "(data/slime_*.jsonl). Use a frozen operator/HF held-out "
                    "JSONL, or omit eval_dataset when data_gen materialize will "
                    "auto-split. Smoke/CI: SEISO_ALLOW_TINY_RL=1."
                )
            if is_slime_ci_fixture_path(self.dataset) and not materialize_will_split:
                raise ValueError(
                    f"dataset={self.dataset} is a slime CI fixture "
                    "(data/slime_*.jsonl) — not a training corpus. Point "
                    "dataset at a verifiable operator JSONL, or enable "
                    "data_gen with data_gen_source=dataset and "
                    "dataset_ref=<HF hub id> (e.g. open-r1/OpenR1-Math-220k). "
                    "Smoke/CI: configs/smoke_slime_cpu.yaml + "
                    "SEISO_ALLOW_TINY_RL=1."
                )
            # Local JSONL must exist (or be replaced by materialize). Shape-only
            # templates (operator_*.jsonl placeholders) need
            # SEISO_ALLOW_TEMPLATE_SLIME=1 until paths point at real files.
            if not allow_template_slime():
                if unresolved_local_jsonl(self.eval_dataset):
                    raise ValueError(
                        f"eval_dataset={self.eval_dataset} is missing on disk. "
                        "Point it at a frozen held-out JSONL, omit it when "
                        "data_gen materialize will auto-split, or set "
                        "SEISO_ALLOW_TEMPLATE_SLIME=1 for shape-only YAMLs."
                    )
                if unresolved_local_jsonl(self.dataset) and not materialize_will_split:
                    raise ValueError(
                        f"dataset={self.dataset} is missing on disk. Point "
                        "dataset at a verifiable JSONL, enable data_gen + "
                        "dataset_ref for Hub materialize, or set "
                        "SEISO_ALLOW_TEMPLATE_SLIME=1 for shape-only YAMLs."
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
        from seiso.security import assert_relative_artifact_name

        assert_relative_artifact_name(self.best_checkpoint_dir, field="best_checkpoint_dir")
        if self.final_checkpoint_dir:
            assert_relative_artifact_name(self.final_checkpoint_dir, field="final_checkpoint_dir")
        # Weight-sync dirs are joined under output_dir during HTTP rollout sync —
        # reject .. / absolute the same way as checkpoint artifact names.
        assert_relative_artifact_name(self.sglang_weight_dir, field="sglang_weight_dir")
        assert_relative_artifact_name(self.vllm_weight_dir, field="vllm_weight_dir")
        if self.write_verifier_data and not self.verifier_data_file:
            raise ValueError("verifier_data_file must not be empty")
        if self.verifier_data_file:
            assert_relative_artifact_name(self.verifier_data_file, field="verifier_data_file")
        if self.data_gen_filename:
            assert_relative_artifact_name(self.data_gen_filename, field="data_gen_filename")
        if self.data_gen_count < 0:
            raise ValueError("data_gen_count must be non-negative")
        if self.data_gen and self.data_gen_count < 1:
            raise ValueError(
                "data_gen requires data_gen_count >= 1 "
                "(prefer 200+ for meaningful GRPO outcome diversity)"
            )
        # Product floors / source readiness — fail at validate, not mid-job.
        materialize_requested = bool(self.data_gen or self.data_gen_count > 0)
        src = _normalize_data_gen_source(str(self.data_gen_source or "off"))
        if materialize_requested and src in {"off", "none"}:
            raise ValueError(
                "data_gen is enabled but data_gen_source is off. Set "
                "data_gen_source=dataset|data_designer|auto, or disable data_gen "
                "and point dataset at a grounded JSONL."
            )
        if materialize_requested and src in {"auto", ""}:
            ref_ready = bool((self.dataset_ref or "").strip()) or dataset_as_hub
            if not ref_ready and not dd_on:
                raise ValueError(
                    "data_gen_source=auto needs dataset_ref (or a Hub dataset "
                    "id) or data_designer=on before train. Otherwise disable "
                    "data_gen and set dataset + eval_dataset JSONL paths."
                )
        if src == "dataset" and materialize_requested:
            ref = (self.dataset_ref or "").strip()
            if not ref and not dataset_as_hub:
                raise ValueError(
                    "data_gen_source=dataset requires dataset_ref (HF hub id) "
                    "or a non-file dataset ref"
                )
        if src == "data_designer" and materialize_requested and dd_off:
            raise ValueError(
                "data_gen_source=data_designer but data_designer=off. Set "
                "data_designer=on with a vLLM/OpenAI endpoint, or use "
                "data_gen_source=dataset / an operator JSONL dataset."
            )
        if materialize_requested and not allow_tiny_rl():
            floor = PRODUCT_DATA_GEN_FLOOR
            if self.require_held_out_eval and self.eval_dataset is None and materialize_will_split:
                floor = min_data_gen_count_for_held_out_split(PRODUCT_DATA_GEN_FLOOR)
            if self.data_gen_count < floor:
                raise ValueError(
                    f"data_gen_count={self.data_gen_count} is below the product "
                    f"grounded floor ({floor}"
                    + (
                        f"; need >= {floor} when held-out auto-splits 10% "
                        f"from materialize so train keeps "
                        f"{PRODUCT_DATA_GEN_FLOOR}"
                        if floor > PRODUCT_DATA_GEN_FLOOR
                        else ""
                    )
                    + "). Raise data_gen_count, set an explicit frozen "
                    "eval_dataset, or set SEISO_ALLOW_TINY_RL=1 for smoke/CI."
                )
        if not self.data_gen_filename:
            raise ValueError("data_gen_filename must not be empty")
        if not self.data_gen_mix:
            raise ValueError("data_gen_mix must not be empty")
        if not self.data_gen_difficulty:
            raise ValueError("data_gen_difficulty must not be empty")
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
