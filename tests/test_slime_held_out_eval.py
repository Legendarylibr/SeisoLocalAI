"""Held-out slime eval: scoring + config validation (no GPU generate)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.eval import load_eval_samples, run_held_out_eval, score_held_out_completions
from seiso.slime.types import _DistributedSlimeContext


def test_ci_fixture_code_eval_is_disjoint_from_train_fixture():
    """Committed code fixtures stay valid for smoke/tests (not product examples)."""
    train = Path("data/slime_code_sample.jsonl")
    eval_path = Path("data/slime_code_eval.jsonl")
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=train,
        output_dir=Path("./.test_outputs/code-eval"),
        eval_dataset=eval_path,
        require_held_out_eval=False,
        reward="code",
        answer_field="answer",
        eval_on_complete=True,
        rollouts_per_prompt=2,
        policy_micro_batch_size=2,
    )
    cfg.validate()
    assert cfg.eval_dataset == eval_path
    assert cfg.eval_dataset != cfg.dataset
    assert cfg.eval_on_complete is True
    samples = load_eval_samples(cfg.eval_dataset)
    assert len(samples) >= 8
    train_ids = {
        json.loads(line).get("prompt_id")
        for line in train.read_text(encoding="utf-8").splitlines()
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


def test_maybe_run_held_out_eval_barriers_all_ranks(tmp_path: Path, monkeypatch):
    """Non-main ranks must wait around main-only generate (DDP safety)."""
    from seiso.slime import trainer as trainer_mod

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "prompt": "Write add(a, b).",
                "tests": ["assert add(1, 2) == 3"],
                "solution": "def add(a, b):\n    return a + b\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        eval_dataset=eval_path,
        eval_on_complete=True,
        rollouts_per_prompt=2,
    )
    barriers: list[int] = []
    monkeypatch.setattr(
        trainer_mod,
        "_distributed_barrier",
        lambda dist_ctx: barriers.append(dist_ctx.rank),
    )
    ran: list[int] = []
    monkeypatch.setattr(
        "seiso.slime.eval.run_held_out_eval",
        lambda **kwargs: ran.append(1) or {"eval_outcome_pass_rate": 1.0},
    )
    monkeypatch.setattr(trainer_mod, "_append_metrics", lambda *a, **k: None)

    rank0 = _DistributedSlimeContext(enabled=True, world_size=2, rank=0)
    rank1 = _DistributedSlimeContext(enabled=True, world_size=2, rank=1)
    for dist in (rank0, rank1):
        trainer_mod._maybe_run_held_out_eval(
            model=object(),
            tokenizer=object(),
            config=cfg,
            torch=object(),
            dist_ctx=dist,
            step=3,
            metrics_path=tmp_path / "metrics.jsonl",
            force=True,
        )

    assert ran == [1]
    # Each rank barriers before and after main-only work.
    assert barriers == [0, 0, 1, 1]


def test_run_held_out_eval_restores_train_mode(tmp_path: Path, monkeypatch):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "prompt": "Write add(a, b).",
                "tests": ["assert add(1, 2) == 3"],
                "solution": "def add(a, b):\n    return a + b\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = SingleGpuSlimeConfig(
        model_id="test/model",
        dataset=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        eval_dataset=eval_path,
        reward="code",
        rollouts_per_prompt=2,
        require_thinking_trace=False,
    )
    modes: list[str] = []

    class _FakeModel:
        def __init__(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False
            modes.append("eval")

        def train(self, mode: bool = True) -> None:
            self.training = bool(mode)
            modes.append("train" if mode else "eval")

    model = _FakeModel()
    monkeypatch.setattr(
        "seiso.slime.eval._generation_model",
        lambda m: m,
    )
    monkeypatch.setattr(
        "seiso.slime.eval.format_generation_prompt",
        lambda *a, **k: "prompt",
    )
    monkeypatch.setattr(
        "seiso.slime.eval.generate_data_gen_chunk",
        lambda **kwargs: SimpleNamespace(
            completions=["```python\ndef add(a, b):\n    return a + b\n```"]
        ),
    )

    metrics = run_held_out_eval(
        model=model,
        tokenizer=MagicMock(),
        config=cfg,
        torch=MagicMock(),
        step=1,
    )
    assert metrics is not None
    assert modes[0] == "eval"
    assert modes[-1] == "train"
    assert model.training is True
