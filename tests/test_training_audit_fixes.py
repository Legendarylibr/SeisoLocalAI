"""Regression tests for training/RL audit remediations."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.training.config import TrainConfig
from seiso.training.trainer import SeisoTrainer


def test_eval_split_ratio_zero_skips_holdout_even_with_early_stopping(tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "data.jsonl",
            "eval_split_ratio": 0,
            "early_stopping": True,
            "quant": "none",
        }
    )
    trainer = SeisoTrainer(cfg)

    class _DS:
        def __len__(self):
            return 100

        def train_test_split(self, *args, **kwargs):
            raise AssertionError("must not split when eval_split_ratio=0")

    train_ds, eval_ds = trainer._split_train_eval(_DS())
    assert eval_ds is None
    assert len(train_ds) == 100


def test_latest_checkpoint_dir_picks_highest_step(tmp_path: Path):
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-12").mkdir()
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-not-a-step").mkdir()
    latest = SeisoTrainer._latest_checkpoint_dir(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-12"


def test_slime_multi_epoch_auto_kl(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SEISO_SLIME_ALLOW_ZERO_KL", raising=False)
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "slime.jsonl",
        output_dir=tmp_path / "out",
        epochs=3,
        kl_coef=0.0,
    )
    cfg.validate()
    assert cfg.kl_coef == pytest.approx(0.02)


def test_slime_allow_zero_kl_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SEISO_SLIME_ALLOW_ZERO_KL", "1")
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "slime.jsonl",
        output_dir=tmp_path / "out",
        epochs=3,
        kl_coef=0.0,
    )
    cfg.validate()
    assert cfg.kl_coef == 0.0


def test_greater_is_better_for_loss_metrics():
    from seiso.training.trainer import greater_is_better_for_metric

    assert greater_is_better_for_metric("eval_loss") is False
    assert greater_is_better_for_metric("loss") is False
    assert greater_is_better_for_metric("train_loss") is False
    assert greater_is_better_for_metric("eval_accuracy") is True
    assert greater_is_better_for_metric("reward") is True


def test_resolve_trust_remote_code_reads_top_level(tmp_path: Path):
    from seiso.training.trainer import resolve_trust_remote_code

    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "data.jsonl",
            "quant": "none",
            "trust_remote_code": True,
        }
    )
    assert resolve_trust_remote_code(cfg) is True
    cfg_extra = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "data.jsonl",
            "quant": "none",
            "trust_remote_code": False,
            "extra": {"trust_remote_code": True},
        }
    )
    assert resolve_trust_remote_code(cfg_extra) is True


def test_slime_rejects_weight_dir_path_escape(tmp_path: Path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "slime.jsonl",
        output_dir=tmp_path / "out",
        sglang_weight_dir="../../../escape",
    )
    with pytest.raises(ValueError, match="sglang_weight_dir|\\.\\."):
        cfg.validate()


def test_train_config_rejects_vllm_weight_dir_escape(tmp_path: Path):
    with pytest.raises(ValueError, match="\\.\\.|artifact"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "data.jsonl",
                "quant": "none",
                "vllm_weight_dir": "../../escape",
            }
        )
