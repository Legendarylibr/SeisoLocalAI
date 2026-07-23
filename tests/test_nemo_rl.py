"""Tests for NVIDIA NeMo RL Seiso wiring (external launcher)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seiso.nemo_rl.bootstrap import nemo_rl_available, resolve_nemo_rl_root
from seiso.nemo_rl.config import NeMoRLConfig, NeMoRLRecipe
from seiso.nemo_rl.config_builder import build_command, build_hydra_overrides
from seiso.nemo_rl.runner import train_nemo_rl
from seiso.training.config import TrainConfig, TrainMethod, run_training
from seiso.training.practices import learning_rate_for_method


def _fake_nemo_tree(root: Path) -> Path:
    (root / "examples" / "configs").mkdir(parents=True)
    (root / "nemo_rl").mkdir(parents=True)
    (root / "examples" / "run_grpo.py").write_text("# stub\n", encoding="utf-8")
    (root / "examples" / "run_dpo.py").write_text("# stub\n", encoding="utf-8")
    (root / "examples" / "run_distillation.py").write_text("# stub\n", encoding="utf-8")
    (root / "examples" / "configs" / "grpo_math_1B.yaml").write_text("grpo: {}\n", encoding="utf-8")
    (root / "examples" / "configs" / "grpo_smoke.yaml").write_text("grpo: {}\n", encoding="utf-8")
    (root / "examples" / "configs" / "dpo.yaml").write_text("dpo: {}\n", encoding="utf-8")
    (root / "examples" / "configs" / "distillation_math.yaml").write_text(
        "distill: {}\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text("[project]\nname='nemo_rl'\n", encoding="utf-8")
    return root


def test_resolve_nemo_rl_root_from_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("SEISO_NEMO_RL_ROOT", raising=False)
    root = _fake_nemo_tree(tmp_path / "RL")
    assert resolve_nemo_rl_root(root) == root.resolve()
    assert nemo_rl_available(root=root)


def test_build_hydra_overrides_grpo():
    cfg = NeMoRLConfig(
        model_id="Qwen/Qwen2.5-1.5B",
        output_dir=Path("outputs/nemo"),
        recipe=NeMoRLRecipe.GRPO,
        gpus_per_node=2,
        max_steps=10,
        rollouts_per_prompt=4,
        num_prompts_per_step=2,
        learning_rate=5e-6,
        use_lora=True,
        extra_overrides=("logger.wandb_enabled=False",),
    )
    overrides = build_hydra_overrides(cfg)
    assert "policy.model_name=Qwen/Qwen2.5-1.5B" in overrides
    assert "cluster.gpus_per_node=2" in overrides
    assert "grpo.max_num_steps=10" in overrides
    assert "grpo.num_generations_per_prompt=4" in overrides
    assert "grpo.num_prompts_per_step=2" in overrides
    assert any(
        o.startswith("policy.optimizer.kwargs.lr=5e-0") for o in overrides
    )
    assert "policy.lora_cfg.enabled=true" in overrides
    assert "logger.wandb_enabled=False" in overrides


def test_build_command_uses_recipe_script(tmp_path):
    root = _fake_nemo_tree(tmp_path / "RL")
    cfg = NeMoRLConfig(
        model_id="m",
        output_dir=tmp_path / "out",
        recipe=NeMoRLRecipe.SMOKE,
        nemo_rl_root=root,
    )
    cmd = build_command(cfg, nemo_root=root, uv="/usr/bin/uv")
    assert cmd[:5] == [
        "/usr/bin/uv",
        "run",
        "python",
        "examples/run_grpo.py",
        "--config",
    ]
    assert cmd[5] == "examples/configs/grpo_smoke.yaml"


def test_train_nemo_rl_dry_run_without_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("SEISO_NEMO_RL_ROOT", raising=False)
    out = tmp_path / "out"
    cfg = NeMoRLConfig(
        model_id="Qwen/Qwen3-0.6B",
        output_dir=out,
        recipe=NeMoRLRecipe.SMOKE,
        dry_run=True,
        max_steps=10,
    )
    result = train_nemo_rl(cfg)
    assert result == out
    assert (out / "nemo_rl_launch.yaml").is_file()
    assert (out / "seiso_manifest.json").is_file()
    manifest = (out / "seiso_manifest.json").read_text(encoding="utf-8")
    assert '"method": "nemo_rl"' in manifest
    assert '"status": "dry_run"' in manifest


def test_train_config_projects_to_nemo_rl(tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model_id": "Qwen/Qwen2.5-1.5B",
            "dataset": str(tmp_path / "data.jsonl"),
            "output_dir": str(tmp_path / "out"),
            "method": "nemo_rl",
            "quant": "none",
            "nemo_rl_recipe": "grpo",
            "nemo_rl_max_steps": 20,
            "nemo_rl_gpus_per_node": 1,
            "nemo_rl_dry_run": True,
            "rollouts_per_prompt": 4,
            "rollout_batch_size": 2,
        }
    )
    assert cfg.method == TrainMethod.NEMO_RL
    nemo = cfg.to_nemo_rl_config()
    assert nemo.recipe == NeMoRLRecipe.GRPO
    assert nemo.max_steps == 20
    assert nemo.rollouts_per_prompt == 4
    assert nemo.num_prompts_per_step == 2
    assert nemo.dry_run is True


def test_run_training_nemo_rl_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("SEISO_NEMO_RL_ROOT", raising=False)
    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt":"1+1","answer":"2"}\n', encoding="utf-8")
    cfg = TrainConfig.model_validate(
        {
            "model_id": "Qwen/Qwen3-0.6B",
            "dataset": str(data),
            "output_dir": str(tmp_path / "out"),
            "method": "nemo_rl",
            "quant": "none",
            "nemo_rl_recipe": "smoke",
            "nemo_rl_dry_run": True,
            "nemo_rl_max_steps": 10,
        }
    )
    out = run_training(cfg)
    assert (out / "seiso_manifest.json").is_file()
    assert (out / "nemo_rl_launch.yaml").is_file()


def test_smoke_yaml_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Copy minimal fields from repo smoke config shape.
    smoke = {
        "model_id": "Qwen/Qwen3-0.6B",
        "dataset": "open-r1/OpenR1-Math-220k",
        "output_dir": "outputs/nemo-rl-smoke",
        "method": "nemo_rl",
        "quant": "none",
        "nemo_rl_recipe": "smoke",
        "nemo_rl_dry_run": True,
        "nemo_rl_max_steps": 10,
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(smoke), encoding="utf-8")
    cfg = TrainConfig.from_yaml(path)
    assert cfg.method == TrainMethod.NEMO_RL
    assert cfg.nemo_rl_dry_run is True


def test_example_nemo_rl_yaml_loads_without_tiny_rl(monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    cfg = TrainConfig.from_yaml("configs/example_training_nemo_rl.yaml")
    assert cfg.method == TrainMethod.NEMO_RL
    assert cfg.rollouts_per_prompt >= 2
    from seiso.slime.config import is_slime_ci_fixture_path

    assert not is_slime_ci_fixture_path(cfg.dataset)


def test_nemo_rl_grpo_refuses_single_rollout(tmp_path):
    with pytest.raises(ValueError, match="rollouts_per_prompt must be >= 2"):
        NeMoRLConfig(
            model_id="m",
            output_dir=tmp_path / "o",
            recipe=NeMoRLRecipe.GRPO,
            rollouts_per_prompt=1,
        ).validate()


def test_nemo_rl_builder_defaults_rollouts_when_unset(tmp_path):
    cfg = NeMoRLConfig(
        model_id="m",
        output_dir=tmp_path / "o",
        recipe=NeMoRLRecipe.SMOKE,
        rollouts_per_prompt=None,
    )
    overrides = build_hydra_overrides(cfg)
    assert "grpo.num_generations_per_prompt=4" in overrides


def test_preference_requires_dpo_recipe(tmp_path):
    with pytest.raises(ValueError, match="nemo_rl_recipe=dpo"):
        TrainConfig.model_validate(
            {
                "model_id": "m",
                "dataset": str(tmp_path / "p.jsonl"),
                "output_dir": str(tmp_path / "o"),
                "method": "nemo_rl",
                "dataset_format": "preference",
                "nemo_rl_recipe": "grpo",
                "quant": "none",
            }
        )


def test_preference_dpo_recipe_ok(tmp_path):
    cfg = TrainConfig.model_validate(
        {
            "model_id": "m",
            "dataset": str(tmp_path / "p.jsonl"),
            "output_dir": str(tmp_path / "o"),
            "method": "nemo_rl",
            "dataset_format": "preference",
            "nemo_rl_recipe": "dpo",
            "quant": "none",
            "nemo_rl_dry_run": True,
        }
    )
    assert cfg.nemo_rl_recipe == "dpo"


def test_learning_rate_for_nemo_rl():
    assert learning_rate_for_method(TrainMethod.NEMO_RL) == 5e-6


def test_cli_registers_nemo_rl():
    from seiso_cli.main import app

    names = {cmd.name for cmd in app.registered_commands}
    assert "nemo-rl" in names
