from __future__ import annotations

import random
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
    _assign_grouped_advantages,
    _chunked,
    _empty_stats,
    _iter_sample_batches,
    _load_samples,
    _merge_stats,
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


def test_single_gpu_slime_defaults_do_not_load_reference_model(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )

    assert cfg.kl_coef == 0.0


def test_example_single_gpu_slime_config_loads_samples():
    cfg = SingleGpuSlimeConfig.from_yaml(Path("configs/example_slime_single_gpu.yaml"))
    samples = list(_load_samples(cfg))

    assert cfg.dataset == Path("data/slime_sample.jsonl")
    assert cfg.kl_coef == 0.0
    assert cfg.policy_micro_batch_size == 2
    assert cfg.shuffle_buffer_size == 128
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
