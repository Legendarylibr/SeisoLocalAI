from __future__ import annotations

import json
import math
import random
import sys
import types
from pathlib import Path

import pytest

from seiso.models.lora_targets import resolve_lora_target_modules
from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.rewards import (
    contains_answer_reward,
    exact_match_reward,
    numeric_reward,
    resolve_reward,
)
from seiso.slime.trainer import (
    Rollout,
    _append_jsonl_records,
    _append_metrics,
    _apply_lora,
    _assign_grouped_advantages,
    _AutoStopController,
    _balanced_rank_samples,
    _bounded_verifier_metadata,
    _check_training_health,
    _chunked,
    _clipped_policy_loss,
    _collect_training_rollout_batch,
    _DistributedSlimeContext,
    _empty_stats,
    _filter_rollout_groups,
    _final_output_dir,
    _format_rollout_prompt,
    _freeze_multimodal_backbones,
    _group_reward_spread_mean,
    _group_verifier_stats,
    _iter_distributed_sample_batches,
    _iter_sample_batches,
    _load_samples,
    _merge_stats,
    _metric_record,
    _process_reward,
    _PushbackIterator,
    _rank_verifier_path,
    _response_mask_for_sequence,
    _reward_sample,
    _rollout_status,
    _rollout_status_stats,
    _sample_metadata,
    _sampling_batch_size,
    _score_completion,
    _split_thinking_trace,
    _truncate_text,
)


def test_single_gpu_slime_config_from_yaml(tmp_path: Path):
    config_path = tmp_path / "slime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_id: test/model",
                "dataset: data/train.jsonl",
                "output_dir: outputs/slime",
                "rollouts_per_prompt: 3",
                "max_vram_gb: 12",
            ]
        ),
        encoding="utf-8",
    )

    cfg = SingleGpuSlimeConfig.from_yaml(config_path)

    assert cfg.model_id == "test/model"
    assert cfg.dataset == Path("data/train.jsonl")
    assert cfg.output_dir == Path("outputs/slime")
    assert cfg.rollouts_per_prompt == 3
    assert cfg.max_vram_gb == 12
    # default rollout_batch_size is prompts (slime); yaml may omit it → default 1
    assert cfg.rollout_batch_size >= 1
    assert cfg.policy_micro_batch_size == 4
    assert cfg.shuffle_buffer_size == 2048
    assert cfg.require_thinking_trace is True
    assert cfg.outcome_reward_weight == 1.0
    assert cfg.format_reward_weight == 0.1
    assert cfg.process_reward_weight == 0.0
    assert cfg.metadata_field == "metadata"
    assert cfg.use_lora is False


def test_single_gpu_slime_defaults_do_not_load_reference_model_or_lora(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )

    assert cfg.kl_coef == 0.0
    assert cfg.use_lora is False
    assert cfg.dynamic_sampling_filter == "reward_nonzero_std"
    assert cfg.process_reward_weight == 0.0


def test_example_single_gpu_slime_config_loads_samples():
    cfg = SingleGpuSlimeConfig.from_yaml(Path("configs/example_slime_single_gpu.yaml"))
    samples = list(_load_samples(cfg))

    assert cfg.dataset == Path("data/slime_sample.jsonl")
    assert cfg.kl_coef == 0.0
    assert cfg.dynamic_sampling_filter == "reward_nonzero_std"
    assert cfg.policy_micro_batch_size == 4
    assert cfg.shuffle_buffer_size == 128
    assert cfg.use_lora is True
    assert cfg.lora_r == 16
    assert cfg.reward == "auto"
    assert cfg.answer_field == "answer"
    assert cfg.rollout_backend == "hf"
    assert cfg.data_gen is False
    assert cfg.data_gen_source == "off"
    assert cfg.process_reward_weight == 0.0
    assert len(samples) >= 16
    assert "prompt" in samples[0]


def test_single_gpu_slime_config_requires_grouped_rollouts(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=1,
    )

    with pytest.raises(ValueError, match="rollouts_per_prompt"):
        cfg.validate()


def test_single_gpu_slime_config_rejects_invalid_vram_cap(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        max_vram_gb=0,
    )

    with pytest.raises(ValueError, match="max_vram_gb"):
        cfg.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rollout_batch_size", 0),
        ("policy_micro_batch_size", 0),
        ("shuffle_buffer_size", 0),
        ("max_samples_per_epoch", 0),
        ("max_grad_norm", 0),
        ("outcome_reward_weight", -0.1),
        ("format_reward_weight", -0.1),
        ("process_reward_weight", -0.1),
        ("missing_thinking_penalty", -0.1),
        ("min_thinking_tokens", -1),
    ],
)
def test_single_gpu_slime_config_rejects_invalid_optimization_knobs(
    tmp_path: Path,
    field: str,
    value: int,
):
    kwargs = {
        "model_id": "test/model",
        "dataset": tmp_path / "data.jsonl",
        "output_dir": tmp_path / "out",
        field: value,
    }
    cfg = SingleGpuSlimeConfig(**kwargs)

    with pytest.raises(ValueError, match=field):
        cfg.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lora_r", 0),
        ("lora_alpha", 0),
        ("lora_dropout", 1),
        ("lora_bias", "bad"),
        ("lora_target_modules", []),
    ],
)
def test_single_gpu_slime_config_rejects_invalid_lora_options(
    tmp_path: Path,
    field: str,
    value,
):
    kwargs = {
        "model_id": "test/model",
        "dataset": tmp_path / "data.jsonl",
        "output_dir": tmp_path / "out",
        "use_lora": True,
        field: value,
    }
    cfg = SingleGpuSlimeConfig(**kwargs)

    with pytest.raises(ValueError, match=field):
        cfg.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("auto_stop_patience", 0),
        ("auto_stop_min_delta", -0.1),
        ("auto_stop_warmup_steps", -1),
        ("verifier_max_text_chars", -1),
        ("best_checkpoint_dir", ""),
        ("verifier_data_file", ""),
        ("save_every_steps", -1),
        ("log_every_steps", 0),
    ],
)
def test_single_gpu_slime_config_rejects_invalid_stability_options(
    tmp_path: Path,
    field: str,
    value,
):
    kwargs = {
        "model_id": "test/model",
        "dataset": tmp_path / "data.jsonl",
        "output_dir": tmp_path / "out",
        field: value,
    }
    cfg = SingleGpuSlimeConfig(**kwargs)

    with pytest.raises(ValueError, match=field):
        cfg.validate()


def test_single_gpu_slime_config_allows_rollout_batch_as_prompt_count(tmp_path: Path):
    """slime: rollout_batch_size is prompts; may be < rollouts_per_prompt."""
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=4,
        rollout_batch_size=2,
    )
    cfg.validate()


def test_single_gpu_slime_config_requires_oversample_ge_rollout_batch(tmp_path: Path):
    # slime: over_sampling_batch_size >= rollout_batch_size (prompts)
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=4,
        rollout_batch_size=8,
        train_batch_size=4,
        over_sampling_batch_size=4,
        dynamic_sampling_filter="reward_nonzero_std",
    )
    with pytest.raises(ValueError, match="over_sampling_batch_size must be >= rollout_batch_size"):
        cfg.validate()


def test_single_gpu_slime_config_allows_oversample_ge_rollout_batch(
    tmp_path: Path,
):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=4,
        rollout_batch_size=2,
        train_batch_size=2,
        over_sampling_batch_size=4,
        dynamic_sampling_filter="reward_nonzero_std",
    )
    cfg.validate()


def test_single_gpu_slime_config_rejects_clip_ratio_high_below_low(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        clip_ratio=0.2,
        clip_ratio_high=0.1,
    )
    with pytest.raises(ValueError, match="clip_ratio_high"):
        cfg.validate()


def test_single_gpu_slime_config_rejects_empty_metadata_field(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        metadata_field="",
    )

    with pytest.raises(ValueError, match="metadata_field"):
        cfg.validate()


def test_sample_batches_stream_with_bounded_shuffle(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"prompt":"p0","answer":"a0"}',
                '{"prompt":"p1","answer":"a1"}',
                '{"prompt":"p2","answer":"a2"}',
                '{"messages":[]}',
            ]
        ),
        encoding="utf-8",
    )
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=path,
        output_dir=tmp_path / "out",
        train_batch_size=2,
        shuffle_buffer_size=2,
        max_samples_per_epoch=3,
    )

    batches = list(_iter_sample_batches(cfg, random.Random(7)))

    assert [len(batch) for batch in batches] == [2, 1]
    assert sum(len(batch) for batch in batches) == 3
    assert all({"prompt", "answer"} <= set(sample) for batch in batches for sample in batch)


def test_reward_helpers():
    sample = {"answer": "42"}

    assert exact_match_reward("42", sample) == 1.0
    assert exact_match_reward("The answer is 42", sample) == 0.0
    assert contains_answer_reward("The answer is 42.", sample) == 1.0
    assert numeric_reward("x = 42.00001", sample) == 1.0
    assert resolve_reward("numeric") is numeric_reward


def test_unknown_reward_names_are_rejected():
    with pytest.raises(ValueError, match="unknown reward"):
        resolve_reward("remote_code")


def test_reward_sample_maps_upstream_style_metadata_key(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        answer_field="label",
        reward="field",
        reward_field="score",
        metadata_field="context",
    )
    sample = {
        "prompt": "p",
        "label": "42",
        "score": "0.75",
        "context": '{"session_id":"s1","tool_code":"search"}',
    }

    mapped = _reward_sample(sample, cfg)

    assert mapped["answer"] == "42"
    assert mapped["reward"] == "0.75"
    assert mapped["metadata"] == {"session_id": "s1", "tool_code": "search"}


def test_sample_metadata_accepts_structured_or_plain_values(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )

    assert _sample_metadata({"metadata": {"turn": 1}}, cfg) == {"turn": 1}
    assert _sample_metadata({"metadata": "not json"}, cfg) == "not json"
    assert _sample_metadata({"metadata": ""}, cfg) is None
    assert _sample_metadata({"other": "x"}, cfg) is None


def test_bounded_verifier_metadata_preserves_small_metadata_and_truncates_large():
    assert _bounded_verifier_metadata({"turn": 1}, 100) == {"turn": 1}

    bounded = _bounded_verifier_metadata({"blob": "x" * 50}, 16)

    assert isinstance(bounded, dict)
    assert set(bounded) == {"_truncated"}
    assert len(bounded["_truncated"]) == 16


def test_clipped_policy_loss_supports_per_token_normalization():
    import torch

    new_logprobs = torch.zeros((2, 3))
    old_logprobs = torch.zeros((2, 3))
    advantages = torch.tensor([[1.0], [3.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    # DeepSeek seq-mean: mean([1, 3]) = 2 (not global token-mean 1.5).
    per_token_loss = _clipped_policy_loss(new_logprobs, old_logprobs, advantages, mask, 0.2, torch)
    token_mean_loss = _clipped_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        mask,
        0.2,
        torch,
        aggregation="token_mean",
    )
    per_sample_loss = _clipped_policy_loss(
        torch.zeros(2),
        torch.zeros(2),
        torch.tensor([1.0, 3.0]),
        torch.ones(2),
        0.2,
        torch,
    )

    assert per_token_loss.item() == pytest.approx(-2.0)
    assert token_mean_loss.item() == pytest.approx(-1.5)
    assert per_sample_loss.item() == -2.0


def test_clipped_policy_loss_supports_asymmetric_clip_high():
    import torch

    # ratio = exp(log(1.5)) = 1.5; low clip 0.2 → max 1.2, high 0.28 → max 1.28
    new_logprobs = torch.tensor([math.log(1.5)])
    old_logprobs = torch.zeros(1)
    advantages = torch.tensor([1.0])
    mask = torch.ones(1)

    loss_sym = _clipped_policy_loss(new_logprobs, old_logprobs, advantages, mask, 0.2, torch)
    loss_asym = _clipped_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        mask,
        0.2,
        torch,
        clip_ratio_high=0.28,
    )
    # min(1.5, 1.2) = 1.2 vs min(1.5, 1.28) = 1.28 → more negative loss for higher high-clip
    assert loss_sym.item() == pytest.approx(-1.2)
    assert loss_asym.item() == pytest.approx(-1.28)
    assert loss_asym.item() < loss_sym.item()


def test_rollout_status_detects_stop_length_and_empty():
    import torch

    assert _rollout_status(torch.tensor([], dtype=torch.long), eos_token_id=2) == "empty"
    assert _rollout_status(torch.tensor([7, 2, 9]), eos_token_id=2) == "stop"
    assert _rollout_status(torch.tensor([7, 8, 9]), eos_token_id=2) == "length"


def test_rollout_status_stats_counts_known_statuses():
    rollouts = [
        Rollout(None, None, None, None, None, reward=0.0, status="stop"),
        Rollout(None, None, None, None, None, reward=0.0, status="length"),
        Rollout(None, None, None, None, None, reward=0.0, status="length"),
        Rollout(None, None, None, None, None, reward=0.0, status="ignored"),
    ]

    assert _rollout_status_stats(rollouts) == {
        "rollout_status_stop": 1.0,
        "rollout_status_length": 2.0,
        "rollout_status_empty": 0.0,
    }


def test_thinking_prompt_is_appended_but_completion_is_not_rewritten(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )

    prompt = _format_rollout_prompt("Solve it.", cfg)

    assert cfg.thinking_instruction in prompt
    assert prompt.endswith("<think>")
    # Scoring path uses raw completions; synthetic tags must not be injected.
    jumped = _score_completion(
        " reasoning without tags",
        {"answer": "42"},
        cfg,
    )
    assert jumped["format_ok"] is False
    assert jumped["thinking_penalty"] == cfg.missing_thinking_penalty


def test_completion_scoring_is_outcome_first_with_format_bonus(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        reward="numeric",
        format_reward_weight=0.1,
        process_reward_weight=0.0,
        missing_thinking_penalty=0.25,
    )

    score = _score_completion(
        "<think>First check the arithmetic.</think>42",
        {"answer": "42"},
        cfg,
    )
    # Prompt already opened <think>; model continues body and closes.
    continued = _score_completion(
        "First check the arithmetic.\n</think>\n42",
        {"answer": "42"},
        cfg,
    )
    jumped = _score_completion("42", {"answer": "42"}, cfg)

    assert score["outcome_reward"] == 1.0
    # "First check the arithmetic" = 4 tokens vs min_thinking_tokens=8 → 0.5
    assert score["format_reward"] == pytest.approx(0.5)
    assert score["process_reward"] == 0.0
    assert score["format_ok"] is True
    assert score["thinking_penalty"] == pytest.approx(0.0)
    assert score["outcome_passed"] is True
    assert score["reward"] == pytest.approx(1.0 + 0.1 * 0.5)
    assert continued["format_ok"] is True
    assert continued["format_reward"] == pytest.approx(0.5)
    assert continued["thinking_penalty"] == pytest.approx(0.0)
    assert continued["final_answer"] == "42"
    assert continued["reward"] == pytest.approx(1.0 + 0.1 * 0.5)
    assert jumped["outcome_reward"] == 1.0
    assert jumped["format_ok"] is False
    assert jumped["thinking_penalty"] == 0.25


def test_experimental_process_reward_only_when_weighted(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        process_reward_weight=0.5,
        min_thinking_tokens=4,
    )

    trace, final, complete = _split_thinking_trace(
        "<think>First check. Actually revise.</think>The answer is 7."
    )

    assert trace == "First check. Actually revise."
    assert final == "The answer is 7."
    assert complete is True
    assert _process_reward(trace, final, cfg) > 0.5
    scored = _score_completion(
        "<think>First check. Actually revise.</think>7",
        {"answer": "7"},
        cfg,
    )
    assert scored["process_reward"] > 0.5


def test_grouped_advantages_are_normalized():
    """Match THUDM/slime: (r - mean) / (unbiased_std + 1e-6)."""
    import math

    rollouts = [
        Rollout(
            None,
            None,
            None,
            None,
            None,
            0.0,
            outcome_reward=0.0,
            outcome_passed=False,
        ),
        Rollout(
            None,
            None,
            None,
            None,
            None,
            1.0,
            outcome_reward=1.0,
            outcome_passed=True,
        ),
        Rollout(
            None,
            None,
            None,
            None,
            None,
            2.0,
            outcome_reward=2.0,
            outcome_passed=True,
        ),
        Rollout(
            None,
            None,
            None,
            None,
            None,
            4.0,
            outcome_reward=4.0,
            outcome_passed=True,
        ),
    ]

    _assign_grouped_advantages(rollouts, group_size=2)

    # Group [0, 1]: mean 0.5, unbiased std = sqrt(0.5)
    scale01 = math.sqrt(0.5) + 1e-6
    # Group [2, 4]: mean 3.0, unbiased std = sqrt(2)
    scale24 = math.sqrt(2.0) + 1e-6
    expected = [
        -0.5 / scale01,
        0.5 / scale01,
        -1.0 / scale24,
        1.0 / scale24,
    ]
    assert [r.advantage for r in rollouts] == pytest.approx(expected)
    assert _group_reward_spread_mean(rollouts, group_size=2) == 1.5
    stats = _group_verifier_stats(rollouts, group_size=2)
    assert stats["group_pass_rate"] == 1.0
    assert stats["group_nonzero_spread_frac"] == 1.0
    assert stats["group_nonzero_outcome_spread_frac"] == 1.0
    assert stats["group_outcome_spread_mean"] == 1.5


def test_grouped_advantages_can_disable_std_normalization():
    rollouts = [
        Rollout(None, None, None, None, None, 0.0),
        Rollout(None, None, None, None, None, 2.0),
    ]
    _assign_grouped_advantages(rollouts, group_size=2, grpo_std_normalization=False)
    assert [r.advantage for r in rollouts] == pytest.approx([-1.0, 1.0])


def test_grouped_advantages_zero_when_rewards_identical():
    rollouts = [
        Rollout(None, None, None, None, None, 1.0),
        Rollout(None, None, None, None, None, 1.0),
        Rollout(None, None, None, None, None, 1.0),
        Rollout(None, None, None, None, None, 1.0),
    ]
    _assign_grouped_advantages(rollouts, group_size=4)
    assert all(r.advantage == 0.0 for r in rollouts)


class Linear:
    pass


class _Other:
    pass


class _DummyModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        yield "", self
        yield from self._modules


def test_lora_target_inference_prefers_common_projection_names():
    model = _DummyModel(
        [
            ("model.layers.0.self_attn.q_proj", _Other()),
            ("model.layers.0.self_attn.v_proj", _Other()),
            ("model.layers.0.mlp.down_proj", _Other()),
            ("lm_head", Linear()),
        ]
    )

    assert resolve_lora_target_modules("test/model", model) == [
        "q_proj",
        "v_proj",
    ]


def test_lora_target_inference_scopes_multimodal_to_language_model():
    model = _DummyModel(
        [
            (
                "model.vision_tower.encoder.layers.0.self_attn.q_proj",
                _Other(),
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj",
                Linear(),
            ),
            (
                "model.language_model.layers.0.self_attn.v_proj",
                Linear(),
            ),
        ]
    )
    model.config = types.SimpleNamespace(model_type="gemma4")

    assert resolve_lora_target_modules("google/gemma-4-E4B-it", model) == (
        r".*language_model\..*\.(q_proj|v_proj)"
    )


class _Tower:
    def __init__(self) -> None:
        self.requires_grad_calls: list[bool] = []

    def requires_grad_(self, enabled: bool):
        self.requires_grad_calls.append(enabled)
        return self


def test_freeze_multimodal_backbones_detects_generic_wrapper():
    vision_tower = _Tower()
    audio_tower = _Tower()
    model = _DummyModel(
        [
            ("model.language_model.layers.0.self_attn.q_proj", Linear()),
            ("model.vision_tower.encoder.layers.0.self_attn.q_proj", Linear()),
            ("model.audio_tower.encoder.layers.0.self_attn.q_proj", Linear()),
        ]
    )
    model.model = types.SimpleNamespace(
        vision_tower=vision_tower,
        audio_tower=audio_tower,
    )
    model.config = types.SimpleNamespace(model_type="brand_new_multimodal")

    _freeze_multimodal_backbones(model)

    assert vision_tower.requires_grad_calls == [False]
    assert audio_tower.requires_grad_calls == [False]


def test_lora_target_inference_honors_configured_modules():
    model = _DummyModel(
        [
            ("model.layers.0.self_attn.q_proj", _Other()),
            ("model.layers.0.self_attn.v_proj", _Other()),
        ]
    )

    assert resolve_lora_target_modules(
        "test/model",
        model,
        configured=["v_proj", "q_proj", "q_proj"],
    ) == [
        "v_proj",
        "q_proj",
    ]


def test_apply_lora_reports_training_extra_when_peft_is_missing(monkeypatch, tmp_path: Path):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "peft":
            raise ImportError("missing peft")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        use_lora=True,
    )

    with pytest.raises(RuntimeError, match=r"\.\[train\]"):
        _apply_lora(_DummyModel([]), cfg)


def test_apply_lora_prepares_model_before_wrapping(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    class TaskType:
        CAUSAL_LM = "CAUSAL_LM"

    class LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class WrappedModel(_DummyModel):
        def gradient_checkpointing_enable(self, **kwargs):
            calls.append("grad_ckpt")
            assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}

        def enable_input_require_grads(self):
            calls.append("input_grads")

    def prepare_model_for_kbit_training(model, *, use_gradient_checkpointing):
        assert use_gradient_checkpointing is False
        calls.append("prepare")
        return model

    def get_peft_model(model, lora_config):
        calls.append("wrap")
        assert lora_config.kwargs["target_modules"] == ["q_proj"]
        return WrappedModel(model._modules)

    peft = types.ModuleType("peft")
    peft.LoraConfig = LoraConfig
    peft.TaskType = TaskType
    peft.get_peft_model = get_peft_model
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    monkeypatch.setitem(sys.modules, "peft", peft)

    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        use_lora=True,
        lora_target_modules=["q_proj"],
        gradient_checkpointing=True,
    )
    model = _DummyModel([("model.layers.0.self_attn.q_proj", _Other())])

    wrapped = _apply_lora(model, cfg)

    assert isinstance(wrapped, WrappedModel)
    assert calls == ["wrap", "grad_ckpt", "input_grads"]


def test_auto_stop_controller_tracks_best_reward_and_plateau():
    controller = _AutoStopController(
        enabled=True,
        metric="reward_mean",
        patience=2,
        min_delta=0.1,
        warmup_steps=0,
    )

    first = controller.update(0, {"reward_mean": 1.0})
    second = controller.update(1, {"reward_mean": 1.05})
    third = controller.update(2, {"reward_mean": 1.06})

    assert first.improved is True
    assert second.should_stop is False
    assert third.should_stop is True
    assert third.reason == "auto_stop:reward_mean_plateau"
    assert controller.best_value == 1.0
    assert controller.best_step == 0


def test_auto_stop_controller_minimizes_loss_metric():
    controller = _AutoStopController(
        enabled=True,
        metric="loss",
        patience=2,
        min_delta=0.01,
        warmup_steps=0,
    )

    assert controller.update(0, {"loss": 3.0}).improved is True
    assert controller.update(1, {"loss": 2.98}).improved is True
    assert controller.best_value == 2.98


def test_training_health_stops_on_nonfinite_stats(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )
    disabled = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        stop_on_nonfinite=False,
    )

    assert _check_training_health({"loss": math.nan}, cfg) == "nonfinite:loss"
    assert _check_training_health({"loss": math.nan}, disabled) is None


def test_verifier_jsonl_helpers_bound_text(tmp_path: Path):
    path = tmp_path / "nested" / "verifier.jsonl"

    _append_jsonl_records(
        path,
        [{"prompt": _truncate_text("abcdef", 3)}, {"prompt": _truncate_text("x", 0)}],
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"prompt": "abc"}, {"prompt": ""}]


def test_final_output_dir_can_be_nested(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        final_checkpoint_dir="checkpoint-final",
    )

    assert _final_output_dir(cfg) == tmp_path / "out" / "checkpoint-final"


def test_chunked_splits_work_for_single_gpu_microbatches():
    assert list(_chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_dynamic_sampling_filter_keeps_reward_diverse_groups(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=2,
        dynamic_sampling_filter="reward_nonzero_std",
        over_sampling_batch_size=4,
    )
    # Filter keys off *outcome_reward*, not composite reward (format-only spread).
    rollouts = [
        Rollout(
            None,
            None,
            None,
            None,
            None,
            reward=1.0,
            outcome_reward=1.0,
            outcome_passed=True,
        ),
        Rollout(
            None,
            None,
            None,
            None,
            None,
            reward=1.0,
            outcome_reward=1.0,
            outcome_passed=True,
        ),
        Rollout(
            None,
            None,
            None,
            None,
            None,
            reward=0.0,
            outcome_reward=0.0,
            outcome_passed=False,
        ),
        Rollout(
            None,
            None,
            None,
            None,
            None,
            reward=1.0,
            outcome_reward=1.0,
            outcome_passed=True,
        ),
    ]

    kept, kept_groups, rejected = _filter_rollout_groups(rollouts, cfg)

    assert kept == rollouts[2:]
    assert kept_groups == {1}
    assert rejected == 1
    assert _sampling_batch_size(cfg) == 4


def test_dynamic_sampling_drops_format_only_spread(tmp_path: Path):
    """Composite reward may differ from format alone; outcomes equal → drop."""
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=2,
        dynamic_sampling_filter="reward_nonzero_std",
    )
    rollouts = [
        # All outcomes 0; format shaping makes composite rewards differ.
        Rollout(
            None,
            None,
            None,
            None,
            None,
            reward=0.1,
            outcome_reward=0.0,
            outcome_passed=False,
            format_ok=True,
        ),
        Rollout(
            None,
            None,
            None,
            None,
            None,
            reward=-0.5,
            outcome_reward=0.0,
            outcome_passed=False,
            format_ok=False,
        ),
    ]
    kept, kept_groups, rejected = _filter_rollout_groups(rollouts, cfg)
    assert kept == []
    assert kept_groups == set()
    assert rejected == 1


def test_over_sampling_is_ignored_without_dynamic_filter(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        train_batch_size=2,
        over_sampling_batch_size=8,
        dynamic_sampling_filter="none",
    )

    assert _sampling_batch_size(cfg) == 2


def test_dynamic_sampling_refills_until_training_group_target(tmp_path: Path, monkeypatch):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        train_batch_size=2,
        rollouts_per_prompt=2,
        dynamic_sampling_filter="reward_nonzero_std",
        over_sampling_batch_size=2,
    )

    def fake_collect_rollouts(**kwargs):
        prompt = kwargs["samples"][0]["prompt"]
        if prompt == "discarded":
            return types.SimpleNamespace(
                rollouts=[],
                stats={
                    "rollout_groups_total": 1.0,
                    "rollout_groups_kept": 0.0,
                    "dynamic_filtered_groups": 1.0,
                },
            )
        return types.SimpleNamespace(
            rollouts=[
                Rollout(None, None, None, None, None, reward=0.0),
                Rollout(None, None, None, None, None, reward=1.0),
            ],
            stats={
                "rollout_groups_total": 1.0,
                "rollout_groups_kept": 1.0,
                "dynamic_filtered_groups": 0.0,
            },
        )

    monkeypatch.setattr(
        "seiso.slime.trainer._collect_rollouts",
        fake_collect_rollouts,
    )
    rollout_batch = _collect_training_rollout_batch(
        model=None,
        ref_model=None,
        tokenizer=None,
        sample_batches=iter([[{"prompt": "kept-1"}], [{"prompt": "kept-2"}]]),
        samples=[{"prompt": "discarded"}],
        config=cfg,
        torch=None,
        epoch=0,
        global_step=0,
        verifier_path=None,
        dist_ctx=_DistributedSlimeContext(enabled=False),
    )

    assert len(rollout_batch.rollouts) == 4
    assert rollout_batch.stats["rollout_groups_total"] == 3.0
    assert rollout_batch.stats["rollout_groups_kept"] == 2.0
    assert rollout_batch.stats["dynamic_filtered_groups"] == 1.0
    assert rollout_batch.stats["dynamic_refill_rounds"] == 2.0
    assert rollout_batch.stats["rollout_groups_target"] == 2.0


def test_dynamic_sampling_truncates_oversampled_groups_to_training_target(
    tmp_path: Path, monkeypatch
):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        train_batch_size=2,
        rollouts_per_prompt=2,
        dynamic_sampling_filter="reward_nonzero_std",
        over_sampling_batch_size=3,
    )

    def fake_collect_rollouts(**_kwargs):
        return types.SimpleNamespace(
            rollouts=[
                Rollout(None, None, None, None, None, reward=0.0),
                Rollout(None, None, None, None, None, reward=1.0),
                Rollout(None, None, None, None, None, reward=0.0),
                Rollout(None, None, None, None, None, reward=1.0),
                Rollout(None, None, None, None, None, reward=0.0),
                Rollout(None, None, None, None, None, reward=1.0),
            ],
            stats={
                "rollout_groups_total": 3.0,
                "rollout_groups_kept": 3.0,
                "dynamic_filtered_groups": 0.0,
            },
        )

    monkeypatch.setattr(
        "seiso.slime.trainer._collect_rollouts",
        fake_collect_rollouts,
    )
    rollout_batch = _collect_training_rollout_batch(
        model=None,
        ref_model=None,
        tokenizer=None,
        sample_batches=iter([]),
        samples=[{"prompt": "p"}],
        config=cfg,
        torch=None,
        epoch=0,
        global_step=0,
        verifier_path=None,
        dist_ctx=_DistributedSlimeContext(enabled=False),
    )

    assert len(rollout_batch.rollouts) == 4
    assert rollout_batch.stats["rollout_groups_kept"] == 3.0
    assert rollout_batch.stats["rollout_groups_target"] == 2.0


def test_distributed_slime_shards_batches_evenly(tmp_path: Path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(json.dumps({"prompt": f"p{i}", "answer": str(i)}) for i in range(8)),
        encoding="utf-8",
    )
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=data,
        output_dir=tmp_path / "out",
        train_batch_size=2,
    )

    class _FakeTorch:
        long = "long"

        @staticmethod
        def tensor(values, **_kwargs):
            return _FakeTensor(values[0])

        class distributed:
            class ReduceOp:
                MIN = "min"

            @staticmethod
            def all_reduce(_tensor, **_kwargs):
                return None

    class _FakeTensor:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    rank0 = _DistributedSlimeContext(enabled=True, world_size=2, rank=0)
    rank1 = _DistributedSlimeContext(enabled=True, world_size=2, rank=1)

    batches0 = list(_iter_distributed_sample_batches(cfg, random.Random(1), rank0, _FakeTorch))
    batches1 = list(_iter_distributed_sample_batches(cfg, random.Random(1), rank1, _FakeTorch))

    assert len(batches0) == len(batches1) == 2
    assert {row["prompt"] for batch in batches0 for row in batch} == {"p0", "p2", "p4", "p6"}
    assert {row["prompt"] for batch in batches1 for row in batch} == {"p1", "p3", "p5", "p7"}


def test_balanced_rank_samples_spreads_prompt_work(tmp_path: Path):
    data = tmp_path / "data.jsonl"
    prompts = ["x" * 100, "x" * 90, "x" * 20, "x" * 10]
    data.write_text(
        "\n".join(json.dumps({"prompt": prompt, "answer": "a"}) for prompt in prompts),
        encoding="utf-8",
    )
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=data,
        output_dir=tmp_path / "out",
        train_batch_size=1,
        balance_data=True,
    )

    rank0 = _DistributedSlimeContext(enabled=True, world_size=2, rank=0)
    rank1 = _DistributedSlimeContext(enabled=True, world_size=2, rank=1)
    samples0 = _balanced_rank_samples(cfg, rank0, random.Random(1))
    samples1 = _balanced_rank_samples(cfg, rank1, random.Random(1))

    load0 = sum(len(sample["prompt"]) for sample in samples0)
    load1 = sum(len(sample["prompt"]) for sample in samples1)
    assert abs(load0 - load1) <= 20
    assert {sample["prompt"] for sample in samples0 + samples1} == set(prompts)


def test_distributed_verifier_path_is_rank_scoped(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        verifier_data_file="slime_verifier_data.jsonl",
    )

    path = _rank_verifier_path(
        cfg,
        _DistributedSlimeContext(enabled=True, world_size=4, rank=2),
    )

    assert path == tmp_path / "out" / "slime_verifier_data.rank2.jsonl"


def test_slime_metrics_emit_training_reward_shape(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("SEISO_EMIT_METRICS_STDOUT", "1")
    path = tmp_path / "metrics.jsonl"

    _append_metrics(path, {"step": 3, "loss": 0.5, "reward_mean": 1.25})

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["type"] == "training"
    assert record["reward"] == 1.25
    assert _metric_record({"reward_mean": 2.0})["reward"] == 2.0
    assert "SEISO_METRIC:" in capsys.readouterr().out


def test_merge_stats_weighted_by_microbatch_size():
    stats = _empty_stats()

    _merge_stats(
        stats,
        {
            "loss": 2.0,
            "policy_loss": 4.0,
            "kl": 1.0,
            "reward_mean": 6.0,
            "reward_max": 7.0,
        },
        weight=0.25,
    )
    _merge_stats(
        stats,
        {
            "loss": 4.0,
            "policy_loss": 8.0,
            "kl": 3.0,
            "reward_mean": 10.0,
            "reward_max": 5.0,
        },
        weight=0.75,
    )

    assert stats["loss"] == 3.5
    assert stats["policy_loss"] == 7.0
    assert stats["kl"] == 2.5
    assert stats["reward_mean"] == 9.0
    assert stats["reward_max"] == 7.0
    assert stats["outcome_reward_mean"] == 0.0
    assert stats["process_reward_mean"] == 0.0


def test_slime_cli_is_registered():
    from seiso_cli.main import app

    names = {command.name for command in app.registered_commands}
    assert "slime" in names


def test_pushback_iterator_requeues_without_dropping():
    source = iter([[{"a": 1}], [{"b": 2}], [{"c": 3}]])
    it = _PushbackIterator(source)
    first = next(it)
    assert first == [{"a": 1}]
    it.push([{"refilled": True}])
    assert next(it) == [{"refilled": True}]
    assert next(it) == [{"b": 2}]
    assert next(it) == [{"c": 3}]
    with pytest.raises(StopIteration):
        next(it)


def test_response_mask_keeps_eos_when_pad_equals_eos():
    torch = pytest.importorskip("torch")
    # prompt=[1,1], response=[10, 2(eos/pad), 2, 2]
    ids = torch.tensor([1, 1, 10, 2, 2, 2])
    mask = _response_mask_for_sequence(
        ids,
        prompt_width=2,
        pad_token_id=2,
        eos_token_id=2,
        torch=torch,
    )
    assert mask.tolist() == [False, False, True, True, False, False]


def test_response_mask_drops_pad_when_pad_differs_from_eos():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([1, 1, 10, 2, 0, 0])
    mask = _response_mask_for_sequence(
        ids,
        prompt_width=2,
        pad_token_id=0,
        eos_token_id=2,
        torch=torch,
    )
    assert mask.tolist() == [False, False, True, True, False, False]
