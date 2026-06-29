from __future__ import annotations

import json
import math
import random
import sys
import types
from pathlib import Path

import pytest

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
    _apply_lora,
    _assign_grouped_advantages,
    _AutoStopController,
    _check_training_health,
    _chunked,
    _empty_stats,
    _final_output_dir,
    _iter_sample_batches,
    _load_samples,
    _merge_stats,
    _resolve_lora_target_modules,
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


def test_grouped_advantages_are_normalized():
    rollouts = [
        Rollout(None, None, None, None, None, 0.0),
        Rollout(None, None, None, None, None, 1.0),
        Rollout(None, None, None, None, None, 2.0),
        Rollout(None, None, None, None, None, 4.0),
    ]

    _assign_grouped_advantages(rollouts, group_size=2)

    assert [r.advantage for r in rollouts] == [-1.0, 1.0, -1.0, 1.0]


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

    assert _resolve_lora_target_modules(model, None) == [
        "q_proj",
        "v_proj",
        "down_proj",
    ]


def test_lora_target_inference_falls_back_to_linear_modules():
    model = _DummyModel(
        [
            ("block.foo", Linear()),
            ("block.bar", Linear()),
            ("lm_head", Linear()),
        ]
    )

    assert _resolve_lora_target_modules(model, None) == ["bar", "foo"]


def test_lora_target_inference_honors_configured_modules():
    model = _DummyModel([])

    assert _resolve_lora_target_modules(model, ["v_proj", "q_proj", "q_proj"]) == [
        "q_proj",
        "v_proj",
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
        def enable_input_require_grads(self):
            calls.append("input_grads")

    def prepare_model_for_kbit_training(model, *, use_gradient_checkpointing):
        assert use_gradient_checkpointing is True
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
    assert calls == ["prepare", "wrap", "input_grads"]


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

    _append_jsonl_records(path, [{"prompt": _truncate_text("abcdef", 3)}, {"prompt": _truncate_text("x", 0)}])

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


def test_merge_stats_weighted_by_microbatch_size():
    stats = _empty_stats()

    _merge_stats(stats, {"loss": 2.0, "policy_loss": 4.0, "kl": 1.0, "reward_mean": 6.0, "reward_max": 7.0}, weight=0.25)
    _merge_stats(stats, {"loss": 4.0, "policy_loss": 8.0, "kl": 3.0, "reward_mean": 10.0, "reward_max": 5.0}, weight=0.75)

    assert stats["loss"] == 3.5
    assert stats["policy_loss"] == 7.0
    assert stats["kl"] == 2.5
    assert stats["reward_mean"] == 9.0
    assert stats["reward_max"] == 7.0


def test_slime_cli_is_registered():
    from seiso_cli.main import app

    names = {command.name for command in app.registered_commands}
    assert "slime" in names
