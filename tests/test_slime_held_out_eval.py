"""Held-out slime eval: scoring + config validation (no GPU generate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.eval import load_eval_samples, score_held_out_completions


def test_example_slime_code_config_has_disjoint_eval_dataset():
    cfg = SingleGpuSlimeConfig.from_yaml(Path("configs/example_slime_code.yaml"))
    cfg.validate()
    assert cfg.eval_dataset == Path("data/slime_code_eval.jsonl")
    assert cfg.eval_dataset != cfg.dataset
    assert cfg.eval_on_complete is True
    samples = load_eval_samples(cfg.eval_dataset)
    assert len(samples) >= 8
    train_ids = {
        json.loads(line).get("prompt_id")
        for line in Path(cfg.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for sample in samples:
        assert sample.get("held_out") is True
        pid = str(sample.get("prompt_id", ""))
        assert pid.startswith("eval_")
        assert pid.removeprefix("eval_") not in train_ids


def test_eval_dataset_must_differ_from_train(tmp_path: Path):
    path = tmp_path / "same.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=path,
        output_dir=tmp_path / "out",
        eval_dataset=path,
        rollouts_per_prompt=2,
    )
    with pytest.raises(ValueError, match="eval_dataset must differ"):
        cfg.validate()


def test_score_held_out_completions_binary_pass_rate(tmp_path: Path):
    eval_path = Path("data/slime_code_eval.jsonl")
    samples = load_eval_samples(eval_path, max_prompts=4)
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        eval_dataset=eval_path,
        reward="code",
        rollouts_per_prompt=2,
        require_thinking_trace=False,
    )
    cfg.validate()
    good = [f"```python\n{s['solution']}```" for s in samples]
    metrics = score_held_out_completions(
        completions=good, samples=samples, config=cfg
    )
    assert metrics["eval_prompt_count"] == 4.0
    assert metrics["eval_outcome_pass_rate"] == 1.0
    assert metrics["eval_proof_pass_rate"] == 1.0

    bad = ["def nope():\n    return None\n"] * len(samples)
    fail = score_held_out_completions(
        completions=bad, samples=samples, config=cfg
    )
    assert fail["eval_outcome_pass_rate"] == 0.0
