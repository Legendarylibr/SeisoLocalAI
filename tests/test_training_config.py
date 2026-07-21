from pathlib import Path

from seiso.training.config import QuantMode, TrainConfig, TrainMethod


def test_train_config_from_dict():
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": "./update.jsonl",
            "method": "lora",
            "quant": "4bit",
        }
    )
    assert cfg.method == TrainMethod.LORA
    assert cfg.quant == QuantMode.INT4
    assert cfg.lora_r == 16
    assert cfg.use_fused_lora is True


def test_train_config_accepts_slime_method():
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": "./slime.jsonl",
            "method": "slime",
            "reward": "contains_answer",
            "max_vram_gb": 12,
            "logging_steps": 1,
        }
    )

    assert cfg.method == TrainMethod.SLIME
    assert cfg.training_methodology == "seiso_release_post_training"
    assert cfg.auto_stop is True
    assert cfg.write_verifier_data is True


def test_train_config_projects_to_single_gpu_slime_config(tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "slime.jsonl",
            "output_dir": tmp_path / "out",
            "method": "slime",
            "metadata_field": "context",
            "reward": "field",
            "reward_field": "score",
            "batch_size": 1,
            "policy_micro_batch_size": 2,
            "rollouts_per_prompt": 3,
            "rollout_batch_size": 6,
            "over_sampling_batch_size": 9,
            "dynamic_sampling_filter": "reward_nonzero_std",
            "dynamic_sampling_min_reward_std": 0.01,
            "clip_ratio": 0.2,
            "clip_ratio_high": 0.28,
            "grpo_std_normalization": True,
            "calculate_per_token_loss": True,
            "balance_data": True,
            "learning_rate": 5e-6,
            "require_thinking_trace": True,
            "format_reward_weight": 0.2,
            "process_reward_weight": 0.0,
            "missing_thinking_penalty": 0.2,
            "min_thinking_tokens": 6,
            "save_steps": 25,
            "logging_steps": 1,
            "extra": {
                "max_steps": 5,
                "lora_target_modules": ["q_proj"],
                "lora_bias": "lora_only",
            },
        }
    )

    slime = cfg.to_single_gpu_slime_config()

    assert slime.model_id == "test/model"
    assert slime.dataset == tmp_path / "slime.jsonl"
    assert slime.output_dir == tmp_path / "out"
    assert slime.reward == "field"
    assert slime.metadata_field == "context"
    assert slime.reward_field == "score"
    # train_batch_size not set → None (effective = rollout_batch_size)
    assert slime.train_batch_size is None
    assert slime.policy_micro_batch_size == 2
    assert slime.rollouts_per_prompt == 3
    assert slime.rollout_batch_size == 6
    assert slime.over_sampling_batch_size == 9
    assert slime.dynamic_sampling_filter == "reward_nonzero_std"
    assert slime.dynamic_sampling_min_reward_std == 0.01
    assert slime.clip_ratio == 0.2
    assert slime.clip_ratio_high == 0.28
    assert slime.grpo_std_normalization is True
    assert slime.calculate_per_token_loss is True
    assert slime.balance_data is True
    assert slime.learning_rate == 5e-6
    assert slime.require_thinking_trace is True
    assert slime.format_reward_weight == 0.2
    assert slime.process_reward_weight == 0.0
    assert slime.missing_thinking_penalty == 0.2
    assert slime.min_thinking_tokens == 6
    assert slime.save_every_steps == 25
    assert slime.log_every_steps == 1
    assert slime.max_steps == 5
    assert slime.use_lora is True
    assert slime.lora_target_modules == ["q_proj"]
    assert slime.lora_bias == "lora_only"


def test_example_training_slime_config_loads():
    cfg = TrainConfig.from_yaml("configs/example_training_slime.yaml")
    slime = cfg.to_single_gpu_slime_config()

    assert cfg.method == TrainMethod.SLIME
    assert slime.reward == "auto"
    assert slime.answer_field == "answer"
    assert slime.rollout_backend == "hf"
    assert slime.data_gen is False
    assert slime.data_gen_source == "off"
    assert slime.eval_dataset is not None
    assert slime.require_held_out_eval is True
    assert slime.process_reward_weight == 0.0
    assert slime.format_reward_weight == 0.1
    assert slime.dynamic_sampling_filter == "reward_nonzero_std"
    assert slime.clip_ratio_high == 0.28
    assert slime.grpo_std_normalization is True
    assert slime.use_lora is True
    assert slime.auto_stop is True


def test_example_slime_yaml_loads_via_train_config_with_aliases():
    """example_slime_* uses slime field names; TrainConfig.from_yaml maps them."""
    cfg = TrainConfig.from_yaml("configs/example_slime_single_gpu.yaml")
    slime = cfg.to_single_gpu_slime_config()

    assert cfg.method == TrainMethod.SLIME
    assert cfg.slime_use_lora is True
    assert cfg.save_steps == 100
    assert cfg.logging_steps == 1
    assert slime.use_lora is True
    assert slime.save_every_steps == 100
    assert slime.log_every_steps == 1
    assert slime.rollout_backend == "hf"


def test_example_slime_code_yaml_maps_held_out_eval_aliases():
    cfg = TrainConfig.from_yaml("configs/example_slime_code.yaml")
    slime = cfg.to_single_gpu_slime_config()

    assert cfg.slime_eval_dataset == Path("data/slime_code_eval.jsonl")
    assert cfg.slime_eval_on_complete is True
    assert slime.eval_dataset == Path("data/slime_code_eval.jsonl")
    assert slime.eval_on_complete is True
    assert slime.eval_dataset != slime.dataset


def test_smoke_and_example_lora_yaml_fields_consumed():
    """Advertised smoke/example LoRA keys round-trip through TrainConfig."""
    for path in ("configs/smoke_train_cpu.yaml", "configs/example_lora.yaml"):
        cfg = TrainConfig.from_yaml(path)
        assert cfg.method == TrainMethod.LORA
        assert cfg.train_on_responses_only is True
        assert cfg.preprocess_dataset is True
        assert cfg.dataset_format.value in ("chat", "auto")
        assert cfg.model_id
        assert str(cfg.dataset)


def test_example_training_slime_ddp_config_loads():
    cfg = TrainConfig.from_yaml("configs/example_training_slime_ddp.yaml")
    slime = cfg.to_single_gpu_slime_config()

    assert cfg.method == TrainMethod.SLIME
    assert cfg.multi_gpu is True
    assert cfg.distributed_strategy.value == "ddp"
    assert cfg.balance_data is True
    assert slime.balance_data is True
    assert slime.data_gen is False
    assert slime.data_gen_source == "off"
    assert slime.eval_dataset is not None
    assert slime.clip_ratio_high == 0.28
    assert slime.grpo_std_normalization is True


def test_train_config_rejects_slime_oversample_below_rollout_batch(tmp_path):
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="over_sampling_batch_size"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "slime.jsonl",
                "output_dir": tmp_path / "out",
                "method": "slime",
                "rollout_batch_size": 8,
                "over_sampling_batch_size": 4,
                "dynamic_sampling_filter": "reward_nonzero_std",
            }
        )


def test_train_config_accepts_legacy_hf_dataset_field_alias(tmp_path: Path):
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "slime.jsonl",
            "output_dir": tmp_path / "out",
            "method": "slime",
            "hf_dataset": "org/math",
            "data_gen_source": "hf_dataset",
        }
    )
    assert cfg.dataset_ref == "org/math"
    assert cfg.data_gen_source == "dataset"
    slime = cfg.to_single_gpu_slime_config()
    assert slime.dataset_ref == "org/math"
    assert slime.data_gen_source == "dataset"


def test_materialize_source_hf_dataset_alias(tmp_path: Path):
    from seiso.rl_verify.synth_materialize import normalize_materialize_source

    assert normalize_materialize_source("hf_dataset") == "dataset"
    assert normalize_materialize_source("dataset") == "dataset"
