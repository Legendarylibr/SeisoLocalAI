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
            "require_held_out_eval": False,
        }
    )

    assert cfg.method == TrainMethod.SLIME
    assert cfg.training_methodology == "seiso_release_post_training"
    assert cfg.auto_stop is True
    assert cfg.write_verifier_data is True


def test_train_config_projects_to_single_gpu_slime_config(tmp_path):
    eval_path = tmp_path / "slime_eval.jsonl"
    eval_path.write_text("{}\n", encoding="utf-8")
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": tmp_path / "slime.jsonl",
            "slime_eval_dataset": eval_path,
            "output_dir": tmp_path / "out",
            "method": "slime",
            "metadata_field": "context",
            "reward": "numeric",
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
    assert slime.reward == "numeric"
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
    assert slime.data_gen is True
    assert slime.data_gen_source == "dataset"
    assert slime.dataset_ref == "open-r1/OpenR1-Math-220k"
    assert slime.data_gen_count >= 256
    assert slime.eval_dataset is None  # materialize auto-splits held-out
    assert slime.require_held_out_eval is True
    assert slime.process_reward_weight == 0.0
    assert slime.format_reward_weight == 0.1
    assert slime.dynamic_sampling_filter == "reward_nonzero_std"
    assert slime.clip_ratio_high == 0.28
    assert slime.grpo_std_normalization is True
    assert slime.use_lora is True
    assert slime.auto_stop is True
    assert slime.policy_micro_batch_size % slime.rollouts_per_prompt == 0


def test_product_slime_examples_load_without_tiny_rl(monkeypatch):
    """Ready-to-run slime YAMLs must validate for operators (no TINY_RL)."""
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    monkeypatch.delenv("SEISO_ALLOW_TEMPLATE_SLIME", raising=False)
    paths = (
        "configs/example_training_slime.yaml",
        "configs/example_slime_single_gpu.yaml",
        "configs/example_training_slime_ddp.yaml",
        "configs/example_training_slime_vllm.yaml",
    )
    for path in paths:
        cfg = TrainConfig.from_yaml(path)
        slime = cfg.to_single_gpu_slime_config()
        slime.validate()
        assert slime.policy_micro_batch_size % slime.rollouts_per_prompt == 0, path
        assert slime.data_gen_count >= 256, path
        from seiso.slime.config import is_slime_ci_fixture_path

        assert not is_slime_ci_fixture_path(slime.dataset), path
        assert not is_slime_ci_fixture_path(slime.eval_dataset), path


def test_product_slime_shape_templates_refuse_missing_paths(monkeypatch):
    """Code/choice examples are shape templates until operator JSONL exists."""
    import pytest
    from pydantic import ValidationError

    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    monkeypatch.delenv("SEISO_ALLOW_TEMPLATE_SLIME", raising=False)
    for path in (
        "configs/example_slime_code.yaml",
        "configs/example_slime_choice.yaml",
    ):
        with pytest.raises(ValidationError, match="missing on disk"):
            TrainConfig.from_yaml(path)


def test_product_slime_shape_templates_load_with_template_escape(monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    monkeypatch.setenv("SEISO_ALLOW_TEMPLATE_SLIME", "1")
    for path in (
        "configs/example_slime_code.yaml",
        "configs/example_slime_choice.yaml",
    ):
        cfg = TrainConfig.from_yaml(path)
        slime = cfg.to_single_gpu_slime_config()
        slime.validate()
        assert slime.policy_micro_batch_size % slime.rollouts_per_prompt == 0


def test_product_training_configs_load_without_tiny_rl(monkeypatch):
    """All advertised example_*.yaml TrainConfigs load without TINY_RL."""
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    monkeypatch.delenv("SEISO_ALLOW_TEMPLATE_SLIME", raising=False)
    # Shape templates need operator files or TEMPLATE escape — covered separately.
    paths = (
        "configs/example_lora.yaml",
        "configs/example_training_slime.yaml",
        "configs/example_slime_single_gpu.yaml",
        "configs/example_training_slime_ddp.yaml",
        "configs/example_training_slime_vllm.yaml",
        "configs/example_training_nemo_rl.yaml",
        "configs/example_training_deterministic.yaml",
    )
    for path in paths:
        cfg = TrainConfig.from_yaml(path)
        from seiso.slime.config import is_slime_ci_fixture_path

        assert not is_slime_ci_fixture_path(cfg.dataset), path
        if cfg.method == TrainMethod.SLIME:
            slime = cfg.to_single_gpu_slime_config()
            assert slime.policy_micro_batch_size % slime.rollouts_per_prompt == 0
            if slime.data_gen or slime.data_gen_count > 0:
                assert slime.data_gen_count >= 256, path
        if cfg.method == TrainMethod.NEMO_RL:
            assert cfg.rollouts_per_prompt >= 2


def test_product_slime_refuses_ci_fixture_dataset(tmp_path, monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="CI fixture"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": "data/slime_sample.jsonl",
                "eval_dataset": "data/slime_numeric_eval.jsonl",
                "output_dir": tmp_path / "out",
                "method": "slime",
                "require_held_out_eval": True,
                "data_gen": False,
            }
        )


def test_product_slime_refuses_sub_floor_data_gen_count(tmp_path, monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="grounded floor"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": "open-r1/OpenR1-Math-220k",
                "output_dir": tmp_path / "out",
                "method": "slime",
                "require_held_out_eval": True,
                "data_gen": True,
                "data_gen_source": "dataset",
                "dataset_ref": "open-r1/OpenR1-Math-220k",
                "data_gen_count": 8,
            }
        )


def test_product_slime_refuses_data_gen_count_that_fails_held_out_split(
    tmp_path, monkeypatch
):
    """256 validates as the corpus floor but fails after 10% auto-split."""
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    import pytest
    from pydantic import ValidationError

    from seiso.slime.config import min_data_gen_count_for_held_out_split

    need = min_data_gen_count_for_held_out_split()
    assert need > 256
    with pytest.raises(ValidationError, match="held-out auto-split|grounded floor"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": "open-r1/OpenR1-Math-220k",
                "output_dir": tmp_path / "out",
                "method": "slime",
                "require_held_out_eval": True,
                "data_gen": True,
                "data_gen_source": "dataset",
                "dataset_ref": "open-r1/OpenR1-Math-220k",
                "data_gen_count": 256,
            }
        )


def test_product_slime_refuses_data_gen_source_off(tmp_path, monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    import pytest

    from seiso.slime.config import SingleGpuSlimeConfig

    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text('{"prompt":"x","answer":"1"}\n', encoding="utf-8")
    eval_path.write_text('{"prompt":"y","answer":"2"}\n', encoding="utf-8")
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=train_path,
        eval_dataset=eval_path,
        output_dir=tmp_path / "out",
        require_held_out_eval=True,
        data_gen=True,
        data_gen_count=300,
        data_gen_source="off",
        rollouts_per_prompt=2,
        policy_micro_batch_size=2,
    )
    with pytest.raises(ValueError, match="data_gen_source is off"):
        cfg.validate()


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


def test_example_slime_code_yaml_maps_held_out_eval_aliases(monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_TEMPLATE_SLIME", "1")
    cfg = TrainConfig.from_yaml("configs/example_slime_code.yaml")
    slime = cfg.to_single_gpu_slime_config()

    assert "operator_code_eval" in str(cfg.slime_eval_dataset)
    assert cfg.slime_eval_on_complete is True
    assert "operator_code_eval" in str(slime.eval_dataset)
    assert slime.eval_on_complete is True
    assert slime.eval_dataset != slime.dataset
    assert "operator_code" in str(slime.dataset)


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
    assert slime.data_gen is True
    assert slime.data_gen_source == "dataset"
    assert slime.dataset_ref == "open-r1/OpenR1-Math-220k"
    assert slime.eval_dataset is None
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
                "require_held_out_eval": False,
            }
        )


def test_train_config_rejects_slime_without_held_out_eval(
    tmp_path: Path, monkeypatch
):
    import pytest
    from pydantic import ValidationError

    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    with pytest.raises(ValidationError, match="eval_dataset is required"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": tmp_path / "slime.jsonl",
                "output_dir": tmp_path / "out",
                "method": "slime",
                "require_held_out_eval": True,
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
            "require_held_out_eval": False,
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


def test_cloud_gpu_slime_vllm_requires_engine_url():
    import pytest

    with pytest.raises(ValueError, match="vllm_base_url"):
        TrainConfig.model_validate(
            {
                "model_id": "test/model",
                "dataset": "./slime.jsonl",
                "method": "slime",
                "require_held_out_eval": False,
                "cloud_gpu_enabled": True,
                "cloud_gpu_provider": "aws",
                "cloud_gpu_instance_type": "p5.48xlarge",
                "rollout_backend": "vllm",
                "vllm_base_url": "",
            }
        )


def test_cloud_gpu_slime_vllm_accepts_engine_url():
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": "./slime.jsonl",
            "method": "slime",
            "require_held_out_eval": False,
            "cloud_gpu_enabled": True,
            "cloud_gpu_provider": "aws",
            "cloud_gpu_instance_type": "p5.48xlarge",
            "rollout_backend": "vllm",
            "vllm_base_url": "http://10.0.0.5:8000",
            "slime_use_lora": True,
            "vllm_weight_mode": "auto",
        }
    )
    assert cfg.vllm_base_url.startswith("http")
    cfg.to_single_gpu_slime_config().validate()
