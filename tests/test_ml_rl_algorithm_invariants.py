"""Math and meaning invariants for SFT / slime GRPO / preference routing."""

from __future__ import annotations

import math

import pytest

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.policy import (
    _assign_grouped_advantages,
    _clipped_policy_loss,
    _kl_k3_from_log_ratio,
)
from seiso.slime.types import Rollout
from seiso.training.config import TrainConfig, TrainMethod


def test_dpo_empty_completion_logps_are_large_negative():
    import torch

    from seiso.adaptive_quant.llm_alignment.dpo_loss import get_batch_logps

    # All labels masked → no completion tokens after causal shift.
    logits = torch.zeros(2, 4, 5)
    labels = torch.full((2, 4), -100)
    logps = get_batch_logps(logits, labels, average_log_prob=False)
    assert torch.all(logps == -1.0e4)


def test_dpo_collator_joint_tokenizes_prompt_completion():
    from seiso.adaptive_quant.llm_alignment.data_collator import DPODataCollator

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, add_special_tokens=True, truncation=True, max_length=64):
            # Character-level encode so joint vs concat diverge on BPE-like merges.
            ids = [ord(c) % 40 + 2 for c in str(text)]
            if add_special_tokens:
                ids = [1] + ids
            return {"input_ids": ids[:max_length]}

    collator = DPODataCollator(tokenizer=_Tok(), max_prompt_length=32, max_length=64)
    encoded = collator._tokenize_pair("ab", "cd")
    joint = _Tok()("abcd", add_special_tokens=True, truncation=True, max_length=64)[
        "input_ids"
    ]
    assert encoded["input_ids"] == joint
    assert encoded["prompt_length"] > 0
    assert all(lab == -100 for lab in encoded["labels"][: encoded["prompt_length"]])
    assert any(lab != -100 for lab in encoded["labels"][encoded["prompt_length"] :])


def test_grounded_reward_source_includes_code():
    from seiso.distill_rl.preferences import _is_grounded_reward_source

    assert _is_grounded_reward_source("verifiable_outcome")
    assert _is_grounded_reward_source("code_proof")
    assert _is_grounded_reward_source("synthetic_code_v1")
    assert not _is_grounded_reward_source("teacher_student")


def test_kl_k3_non_negative_and_zero_at_identity():
    import torch

    zero = torch.zeros(4)
    assert float(_kl_k3_from_log_ratio(zero, torch).item()) == pytest.approx(0.0)

    positive = torch.tensor([0.5, -0.5, 1.0, -2.0])
    value = float(_kl_k3_from_log_ratio(positive, torch).item())
    assert value >= 0.0
    # Hand check: mean(exp(δ)-δ-1) for these δ
    expected = sum(math.exp(d) - d - 1.0 for d in [0.5, -0.5, 1.0, -2.0]) / 4.0
    assert value == pytest.approx(expected, rel=1e-5)


def test_kl_k3_masked_ignores_padding():
    import torch

    log_ratio = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    value = float(_kl_k3_from_log_ratio(log_ratio, torch, mask=mask).item())
    expected = (
        (math.exp(1.0) - 1.0 - 1.0) + (math.exp(3.0) - 3.0 - 1.0)
    ) / 2.0
    assert value == pytest.approx(expected, rel=1e-5)


def test_grouped_advantages_mean_zero_per_group():
    rollouts = [
        Rollout(None, None, None, None, None, 0.0),
        Rollout(None, None, None, None, None, 2.0),
        Rollout(None, None, None, None, None, 1.0),
        Rollout(None, None, None, None, None, 5.0),
    ]
    _assign_grouped_advantages(rollouts, group_size=2, grpo_std_normalization=False)
    assert sum(r.advantage for r in rollouts[:2]) == pytest.approx(0.0)
    assert sum(r.advantage for r in rollouts[2:]) == pytest.approx(0.0)


def test_grouped_advantages_reject_incomplete_trailing_group():
    rollouts = [
        Rollout(None, None, None, None, None, 0.0),
        Rollout(None, None, None, None, None, 1.0),
        Rollout(None, None, None, None, None, 2.0),
    ]
    with pytest.raises(ValueError, match="not divisible"):
        _assign_grouped_advantages(rollouts, group_size=2)


def test_length_normalized_sequence_ratio_stable_vs_token_repeat():
    """Equal mean log-prob shift must not explode with longer sequences."""
    import torch

    advantages = torch.tensor([1.0, 1.0])
    # Short vs long sequences with the same per-token Δlogπ = log(1.1)
    short_new = torch.tensor([math.log(1.1) * 2])
    short_old = torch.zeros(1)
    long_new = torch.tensor([math.log(1.1) * 20])
    long_old = torch.zeros(1)
    # Length-normalized inputs (as policy does when per-token is False)
    short_loss = _clipped_policy_loss(
        short_new / 2.0,
        short_old / 2.0,
        advantages[:1],
        torch.ones(1),
        0.2,
        torch,
    )
    long_loss = _clipped_policy_loss(
        long_new / 20.0,
        long_old / 20.0,
        advantages[:1],
        torch.ones(1),
        0.2,
        torch,
    )
    assert float(short_loss.item()) == pytest.approx(float(long_loss.item()), rel=1e-5)


def test_slime_defaults_per_token_and_outcome_dominant(tmp_path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )
    assert cfg.calculate_per_token_loss is True
    assert cfg.outcome_reward_weight == 1.0
    assert cfg.format_reward_weight == 0.1
    assert cfg.process_reward_weight == 0.0
    # Prefer format bonus over subtractive thinking penalty.
    assert cfg.missing_thinking_penalty == 0.0
    cfg.validate()


def test_slime_rejects_format_dominated_rewards(tmp_path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        outcome_reward_weight=0.1,
        format_reward_weight=0.5,
        process_reward_weight=0.0,
    )
    with pytest.raises(ValueError, match="outcome must dominate"):
        cfg.validate()


def test_slime_rejects_zero_outcome_weight(tmp_path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        outcome_reward_weight=0.0,
    )
    with pytest.raises(ValueError, match="outcome_reward_weight"):
        cfg.validate()


def test_slime_code_reward_mode_defaults_binary_and_rejects_unknown(tmp_path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
    )
    assert cfg.code_reward_mode == "binary"
    cfg.validate()

    bad = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        code_reward_mode="soft",
    )
    with pytest.raises(ValueError, match="code_reward_mode"):
        bad.validate()


def test_auto_code_reward_promotes_to_binary_when_group_has_passer():
    from seiso.slime.config import SingleGpuSlimeConfig
    from seiso.slime.trainer import _finalize_auto_code_rewards
    from seiso.slime.types import Rollout

    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset="unused.jsonl",
        output_dir="unused",
        code_reward_mode="auto",
        rollouts_per_prompt=2,
        outcome_reward_weight=1.0,
        format_reward_weight=0.0,
        process_reward_weight=0.0,
        require_thinking_trace=False,
    )
    # No full passer yet → keep dense.
    dense_group = [
        Rollout(None, None, None, None, None, 0.5, outcome_reward=0.5, proof_score=0.5, proof_passed=False),
        Rollout(None, None, None, None, None, 0.0, outcome_reward=0.0, proof_score=0.0, proof_passed=False),
    ]
    _finalize_auto_code_rewards(dense_group, cfg)
    assert dense_group[0].outcome_reward == pytest.approx(0.5)

    # Full passer present → binary for the whole group.
    mixed = [
        Rollout(None, None, None, None, None, 0.5, outcome_reward=0.5, proof_score=0.5, proof_passed=False),
        Rollout(None, None, None, None, None, 1.0, outcome_reward=1.0, proof_score=1.0, proof_passed=True),
    ]
    _finalize_auto_code_rewards(mixed, cfg)
    assert mixed[0].outcome_reward == 0.0
    assert mixed[1].outcome_reward == 1.0
    assert mixed[0].reward == 0.0
    assert mixed[1].reward == 1.0


def test_auto_code_reward_skips_length_truncated_rollouts():
    from seiso.slime.config import SingleGpuSlimeConfig
    from seiso.slime.trainer import _finalize_auto_code_rewards
    from seiso.slime.types import Rollout

    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset="unused.jsonl",
        output_dir="unused",
        code_reward_mode="auto",
        rollouts_per_prompt=2,
        outcome_reward_weight=1.0,
        format_reward_weight=0.0,
        process_reward_weight=0.0,
        require_thinking_trace=False,
    )
    # Length "passer" must not flip the group to binary; sibling keeps dense score.
    group = [
        Rollout(
            None,
            None,
            None,
            None,
            None,
            0.0,
            outcome_reward=0.0,
            proof_score=1.0,
            proof_passed=True,
            status="length",
        ),
        Rollout(
            None,
            None,
            None,
            None,
            None,
            0.5,
            outcome_reward=0.5,
            proof_score=0.5,
            proof_passed=False,
            status="ok",
        ),
    ]
    _finalize_auto_code_rewards(group, cfg)
    assert group[0].reward == 0.0
    assert group[0].outcome_reward == 0.0
    assert group[1].outcome_reward == pytest.approx(0.5)
    assert group[1].reward == pytest.approx(0.5)


def test_slime_rejects_format_penalty_dominating_outcome(tmp_path):
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        require_thinking_trace=True,
        outcome_reward_weight=1.0,
        missing_thinking_penalty=1.0,
    )
    with pytest.raises(ValueError, match="missing_thinking_penalty"):
        cfg.validate()


def test_slime_rejects_penalty_that_loses_to_format_shaping(tmp_path):
    """penalty > outcome - format allows wrong+formatted to beat correct+unformatted."""
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        require_thinking_trace=True,
        outcome_reward_weight=1.0,
        format_reward_weight=0.5,
        process_reward_weight=0.0,
        missing_thinking_penalty=0.6,
    )
    with pytest.raises(ValueError, match="missing_thinking_penalty"):
        cfg.validate()


def test_train_config_rejects_penalty_that_loses_to_format_shaping(tmp_path):
    with pytest.raises(ValueError, match="missing_thinking_penalty"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "prefs.jsonl",
                "method": "slime",
                "require_thinking_trace": True,
                "outcome_reward_weight": 1.0,
                "format_reward_weight": 0.5,
                "process_reward_weight": 0.0,
                "missing_thinking_penalty": 0.6,
            }
        )


def test_train_config_refuses_preference_without_opt_in(tmp_path):
    with pytest.raises(ValueError, match="Preference datasets"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "prefs.jsonl",
                "dataset_format": "preference",
                "preference_as_sft": False,
            }
        )


def test_train_config_refuses_preference_with_slime(tmp_path):
    with pytest.raises(ValueError, match="incompatible with method=slime"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "prefs.jsonl",
                "dataset_format": "preference",
                "preference_as_sft": True,
                "method": "slime",
            }
        )


def test_train_config_allows_preference_as_sft(tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "prefs.jsonl",
            "dataset_format": "preference",
            "preference_as_sft": True,
        }
    )
    assert cfg.preference_as_sft is True


def test_train_config_rejects_packing_with_response_mask_chat(tmp_path):
    with pytest.raises(ValueError, match="packing cannot be combined"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "chat.jsonl",
                "dataset_format": "chat",
                "packing": True,
                "train_on_responses_only": True,
            }
        )


def test_train_config_rejects_packing_with_response_mask_auto(tmp_path):
    with pytest.raises(ValueError, match="packing cannot be combined"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "auto.jsonl",
                "dataset_format": "auto",
                "packing": True,
                "train_on_responses_only": True,
            }
        )


def test_train_config_allows_packing_with_response_mask_on_text(tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "text.jsonl",
            "dataset_format": "text",
            "packing": True,
            "train_on_responses_only": True,
        }
    )
    assert cfg.packing is True


def test_train_config_rejects_full_with_int4(tmp_path):
    with pytest.raises(ValueError, match="method=full cannot use quant"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "data.jsonl",
                "method": "full",
                "quant": "4bit",
            }
        )


def test_train_config_slime_projection_defaults_per_token(tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "slime.jsonl",
            "method": "slime",
            "epochs": 1,
        }
    )
    assert cfg.calculate_per_token_loss is True
    assert cfg.method == TrainMethod.SLIME
    assert cfg.kl_coef == 0.0
    slime = cfg.to_single_gpu_slime_config()
    assert slime.calculate_per_token_loss is True
    assert slime.outcome_reward_weight > 0
    assert (
        slime.format_reward_weight + slime.process_reward_weight
        <= slime.outcome_reward_weight
    )


def test_train_config_slime_multi_epoch_applies_kl_coef(tmp_path, monkeypatch):
    monkeypatch.delenv("SEISO_SLIME_ALLOW_ZERO_KL", raising=False)
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "slime.jsonl",
            "method": "slime",
            "epochs": 3,
            "kl_coef": 0.0,
        }
    )
    assert cfg.kl_coef == pytest.approx(0.02)


def test_dpo_uses_sum_logps_by_default():
    from seiso.adaptive_quant.llm_alignment.config import DPOSettings
    from seiso.distill_rl.config import DistillRLConfig

    settings = DPOSettings()
    assert settings.average_log_prob is False
    assert DistillRLConfig.model_fields["dpo_average_log_prob"].default is False


def test_recommendation_evidence_simulator_not_deploy_claimable():
    from seiso.rl_quant.recommendation import recommendation_evidence

    meta = recommendation_evidence(
        {
            "evidence_level": "simulator",
            "deploy_quality_claimable": False,
        }
    )
    assert meta["evidence_level"] == "simulator"
    assert meta["deploy_quality_claimable"] is False
    assert meta["deploy_quality_note"]
