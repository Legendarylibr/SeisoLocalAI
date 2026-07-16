"""Tests for high-level verifiable RL data generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seiso.rl_verify.data_gen import (
    DataGenConfig,
    generate_rl_corpus,
    materialize_rl_corpus,
    parse_weight_mix,
)
from seiso.rl_verify.verify import score_completion
from seiso.slime_single_gpu.config import SingleGpuSlimeConfig
from seiso.slime_single_gpu.trainer import _maybe_materialize_data_gen


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


def test_generate_rl_corpus_is_deterministic_and_diverse(tmp_path: Path):
    cfg = DataGenConfig(
        count=60,
        seed=7,
        mix="numeric:0.6,choice:0.4,code:0.0",
        difficulty="easy:0.4,medium:0.4,hard:0.2",
        require_thinking_trace=True,
        verify_code=False,
    )
    a = generate_rl_corpus(cfg)
    b = generate_rl_corpus(cfg)
    assert a.count == 60
    assert [r["task_id"] for r in a.rows] == [r["task_id"] for r in b.rows]
    assert a.summary()["answer_diversity"] > 0.3
    assert a.stream_counts.get("numeric", 0) > 0
    assert a.stream_counts.get("choice", 0) > 0
    # Thinking instruction present for outcome-first slime defaults.
    assert all("<think>" in r["prompt"].lower() or "think" in r["prompt"].lower() for r in a.rows)


def test_numeric_rows_verify_with_correct_answer():
    cfg = DataGenConfig(
        count=20,
        seed=3,
        mix="numeric:1.0",
        difficulty="easy:0.5,medium:0.5,hard:0.0",
        require_thinking_trace=False,
    )
    result = generate_rl_corpus(cfg)
    for row in result.rows:
        answer = str(row["answer"])
        # Simulate a model that emits the gold answer after a closed think block.
        completion = f"<think>calc</think>\n{answer}"
        scored = score_completion(
            completion,
            row,
            checker="numeric",
            require_thinking_trace=False,
            process_weight=0.0,
        )
        assert scored.passed, (row["prompt"], answer, scored)


def test_materialize_writes_jsonl_and_manifest(tmp_path: Path):
    out = tmp_path / "corpus.jsonl"
    result = materialize_rl_corpus(
        out,
        DataGenConfig(
            count=30,
            seed=1,
            mix="numeric:0.7,choice:0.3,code:0.0",
            verify_code=False,
            require_thinking_trace=False,
        ),
    )
    assert out.is_file()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == result.count == 30
    assert json.loads(lines[0])["prompt"]
    manifest = tmp_path / "corpus.manifest.json"
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["count"] == 30


def test_slime_config_data_gen_validation():
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=Path("data/x.jsonl"),
        output_dir=Path("out"),
        data_gen=True,
        data_gen_count=0,
    )
    with pytest.raises(ValueError, match="data_gen_count"):
        cfg.validate()


def test_maybe_materialize_data_gen_rewrites_dataset(tmp_path: Path):
    from seiso.slime_single_gpu.trainer import _DistributedSlimeContext

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
    )
    cfg.validate()
    dist = _DistributedSlimeContext(enabled=False, world_size=1, rank=0, local_rank=0, device="cpu")
    updated = _maybe_materialize_data_gen(cfg, dist)
    assert updated.dataset == tmp_path / "run" / "gen.jsonl"
    assert updated.dataset.is_file()
    n = sum(1 for _ in updated.dataset.open(encoding="utf-8"))
    assert n == 24
