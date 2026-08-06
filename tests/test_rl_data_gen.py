"""Tests for grounded RL materialize helpers (no toy corpus generator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seiso.rl_verify.data_gen import parse_weight_mix, to_slime_prompt_row, write_jsonl
from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.trainer import _maybe_materialize_data_gen


def test_parse_weight_mix_normalizes_and_rejects_unknown():
    mix = parse_weight_mix(
        "numeric:1,choice:1",
        allowed=frozenset({"numeric", "choice", "code"}),
        default={"numeric": 1.0},
    )
    assert mix == pytest.approx({"numeric": 0.5, "choice": 0.5})
    with pytest.raises(ValueError, match="unknown"):
        parse_weight_mix(
            "bogus:1",
            allowed=frozenset({"numeric"}),
            default={"numeric": 1.0},
        )


def test_to_slime_prompt_row_shape():
    row = to_slime_prompt_row(
        "What is 2+2?",
        "4",
        rm_type="numeric",
        metadata={"task_id": "n0"},
    )
    assert row["label"] == "4"
    assert row["answer"] == "4"
    assert row["reward"] == "numeric"
    assert isinstance(row["prompt"], list)
    assert row["prompt"][0]["content"] == "What is 2+2?"


def test_write_jsonl(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    n = write_jsonl(
        path,
        [
            to_slime_prompt_row("q", "1", rm_type="numeric"),
            to_slime_prompt_row("q2", "2", rm_type="numeric"),
        ],
    )
    assert n == 2
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["label"] == "1"


def test_slime_config_data_gen_validation():
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=Path("data/x.jsonl"),
        output_dir=Path("out"),
        data_gen=True,
        data_gen_count=0,
        require_held_out_eval=False,
    )
    with pytest.raises(ValueError, match="data_gen_count"):
        cfg.validate()


def test_maybe_materialize_data_gen_rewrites_dataset(tmp_path: Path, monkeypatch):
    from seiso.rl_verify.data_gen import DataGenResult
    from seiso.slime.trainer import _DistributedSlimeContext

    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=tmp_path / "placeholder.jsonl",
        output_dir=tmp_path / "run",
        data_gen=True,
        data_gen_count=24,
        data_gen_seed=0,
        data_gen_mix="numeric:0.8,choice:0.2,code:0.0",
        data_gen_filename="gen.jsonl",
        require_thinking_trace=False,
        data_gen_source="data_designer",
        data_designer="on",
        vllm_base_url="http://127.0.0.1:8000",
        require_held_out_eval=False,
    )
    cfg.validate()
    dist = _DistributedSlimeContext(enabled=False, world_size=1, rank=0, local_rank=0, device="cpu")
    fake = DataGenResult(
        rows=[{"prompt": "x", "label": "1"} for _ in range(24)],
        stream_counts={"numeric": 24},
        difficulty_counts={},
        seed=0,
    )

    def _materialize(config, *, out_path, count, world_size=1):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "\n".join(json.dumps(r) for r in fake.rows) + "\n",
            encoding="utf-8",
        )
        return fake

    monkeypatch.setattr(
        "seiso.rl_verify.data_designer_gen.data_designer_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "seiso.rl_verify.data_designer_gen.should_use_data_designer",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "seiso.rl_verify.data_designer_gen.materialize_for_slime_config",
        _materialize,
    )
    updated = _maybe_materialize_data_gen(cfg, dist)
    assert updated.dataset == tmp_path / "run" / "gen.jsonl"
    assert updated.dataset.is_file()
    n = sum(1 for _ in updated.dataset.open(encoding="utf-8"))
    assert n == 24
    summary = json.loads(
        (tmp_path / "run" / "slime_data_gen_summary.json").read_text(encoding="utf-8")
    )
    assert summary["generator"] == "nvidia.nemo.data_designer"


def test_toy_generator_api_removed():
    import seiso.rl_verify.data_gen as dg

    assert not hasattr(dg, "generate_rl_corpus")
    assert not hasattr(dg, "materialize_rl_corpus")
    assert not hasattr(dg, "DataGenConfig")
