from __future__ import annotations

import json
import math
import random
import sys
import types
from pathlib import Path

import pytest

from seiso.models.lora_targets import resolve_lora_target_modules
from seiso.slime_single_gpu.config import SingleGpuSlimeConfig
from seiso.slime_single_gpu.rewards import (
    contains_answer_reward,
    exact_match_reward,
    numeric_reward,
    resolve_reward,
)
from seiso.slime_single_gpu.trainer import (
    Rollout,
    _append_jsonl_records,
    _append_metrics,
    _apply_lora,
    _assign_grouped_advantages,
    _AutoStopController,
    _check_training_health,
    _chunked,
    _DistributedSlimeContext,
    _empty_stats,
    _final_output_dir,
    _force_completion_thinking_prefix,
    _format_rollout_prompt,
    _freeze_multimodal_backbones,
    _group_reward_spread_mean,
    _iter_distributed_sample_batches,
    _iter_sample_batches,
    _load_samples,
    _merge_stats,
    _metric_record,
    _process_reward,
    _rank_verifier_path,
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
    assert cfg.rollout_batch_size == 4
    assert cfg.policy_micro_batch_size == 4
    assert cfg.shuffle_buffer_size == 2048
    assert cfg.require_thinking_trace is True
    assert cfg.outcome_reward_weight == 1.0
    assert cfg.process_reward_weight == 0.25
    assert cfg.use_lora is False


def test_single_gpu_slime_defaults_do_not_load_reference_model_or_lora(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )

    assert cfg.kl_coef == 0.0
    assert cfg.use_lora is False


def test_example_single_gpu_slime_config_loads_samples():
    cfg = SingleGpuSlimeConfig.from_yaml(Path("configs/example_slime_single_gpu.yaml"))
    samples = list(_load_samples(cfg))

    assert cfg.dataset == Path("data/slime_sample.jsonl")
    assert cfg.kl_coef == 0.0
    assert cfg.policy_micro_batch_size == 2
    assert cfg.shuffle_buffer_size == 128
    assert cfg.use_lora is True
    assert cfg.lora_r == 16
    assert samples
    assert {"prompt", "answer"} <= set(samples[0])


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


def test_single_gpu_slime_config_requires_rollout_batch_to_cover_group(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=4,
        rollout_batch_size=2,
    )

    with pytest.raises(ValueError, match="rollout_batch_size"):
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


def test_thinking_prompt_and_completion_are_forced(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )

    prompt = _format_rollout_prompt("Solve it.", cfg)

    assert cfg.thinking_instruction in prompt
    assert prompt.endswith("<think>")
    assert _force_completion_thinking_prefix(" reasoning", cfg).startswith("<think>")


def test_completion_scoring_combines_outcome_and_process(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        process_reward_weight=0.5,
        missing_thinking_penalty=0.25,
        min_thinking_tokens=4,
    )

    score = _score_completion(
        "<think>First check the arithmetic because it matters. Therefore 40+2.</think>42",
        {"answer": "42"},
        cfg,
        contains_answer_reward,
    )
    jumped = _score_completion("42", {"answer": "42"}, cfg, contains_answer_reward)

    assert score["outcome_reward"] == 1.0
    assert score["process_reward"] > 0.5
    assert score["thinking_penalty"] == 0.0
    assert score["reward"] > 1.0
    assert jumped["outcome_reward"] == 1.0
    assert jumped["process_reward"] == 0.0
    assert jumped["thinking_penalty"] == 0.25


def test_thinking_trace_split_and_process_reward(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        min_thinking_tokens=4,
    )

    trace, final, complete = _split_thinking_trace(
        "<think>First check. Actually revise.</think>The answer is 7."
    )

    assert trace == "First check. Actually revise."
    assert final == "The answer is 7."
    assert complete is True
    assert _process_reward(trace, final, cfg) > 0.5


def test_grouped_advantages_are_normalized():
    rollouts = [
        Rollout(None, None, None, None, None, 0.0),
        Rollout(None, None, None, None, None, 1.0),
        Rollout(None, None, None, None, None, 2.0),
        Rollout(None, None, None, None, None, 4.0),
    ]

    _assign_grouped_advantages(rollouts, group_size=2)

    assert [r.advantage for r in rollouts] == [-1.0, 1.0, -1.0, 1.0]
    assert _group_reward_spread_mean(rollouts, group_size=2) == 1.5


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
