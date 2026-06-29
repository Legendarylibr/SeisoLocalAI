from __future__ import annotations

from pathlib import Path

import pytest

from seiso.slime_single_gpu.config import SingleGpuSlimeConfig
from seiso.slime_single_gpu.rewards import (
    contains_answer_reward,
    exact_match_reward,
    numeric_reward,
    resolve_reward,
)
from seiso.slime_single_gpu.trainer import Rollout, _assign_grouped_advantages


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


def test_single_gpu_slime_config_requires_grouped_rollouts(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=1,
    )

    with pytest.raises(ValueError, match="rollouts_per_prompt"):
        cfg.validate()


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
        Rollout({}, "p", "a", None, None, None, None, None, 0.0),
        Rollout({}, "p", "b", None, None, None, None, None, 1.0),
        Rollout({}, "q", "c", None, None, None, None, None, 2.0),
        Rollout({}, "q", "d", None, None, None, None, None, 4.0),
    ]

    _assign_grouped_advantages(rollouts, group_size=2)

    assert [r.advantage for r in rollouts] == [-1.0, 1.0, -1.0, 1.0]


def test_slime_cli_is_registered():
    from seiso_cli.main import app

    names = {command.name for command in app.registered_commands}
    assert "slime" in names
