"""Tests for NVIDIA NeMo Data Designer gate + slime row mapping."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from seiso.rl_verify.data_designer_gen import (
    ensure_openai_v1_endpoint,
    is_multigpu_vllm_run,
    normalize_data_designer_mode,
    records_to_slime_rows,
    should_use_data_designer,
)
from seiso.slime.config import SingleGpuSlimeConfig


def _cfg(tmp_path: Path, **kwargs) -> SingleGpuSlimeConfig:
    base = dict(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=2,
        rollout_batch_size=2,
        rollout_backend="vllm",
        vllm_base_url="http://127.0.0.1:8000",
        data_designer="auto",
        require_held_out_eval=False,
    )
    base.update(kwargs)
    return SingleGpuSlimeConfig(**base)


def test_normalize_data_designer_mode():
    assert normalize_data_designer_mode("auto") == "auto"
    assert normalize_data_designer_mode("on") == "on"
    assert normalize_data_designer_mode("OFF") == "off"
    assert normalize_data_designer_mode(None) == "auto"


def test_ensure_openai_v1_endpoint():
    assert ensure_openai_v1_endpoint("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
    assert ensure_openai_v1_endpoint("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    assert ensure_openai_v1_endpoint("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/v1"
    with pytest.raises(ValueError, match="http"):
        ensure_openai_v1_endpoint("file:///tmp")


def test_gate_off_always(tmp_path: Path):
    cfg = _cfg(tmp_path, data_designer="off")
    assert should_use_data_designer(cfg, world_size=8) is False


def test_gate_auto_does_not_silently_select(tmp_path: Path):
    """auto never selects DD; require data_designer=on or data_gen_source=data_designer."""
    cfg = _cfg(tmp_path, data_designer="auto", rollout_backend="hf", vllm_base_url="")
    with patch(
        "seiso.rl_verify.data_designer_gen.data_designer_available",
        return_value=True,
    ):
        assert should_use_data_designer(cfg, world_size=1) is False
    with patch(
        "seiso.rl_verify.data_designer_gen.data_designer_available",
        return_value=False,
    ):
        assert should_use_data_designer(cfg, world_size=8) is False


def test_gate_on_ignores_availability_check(tmp_path: Path):
    """Force-on selects the DD path; materialize still requires the package."""
    cfg = _cfg(tmp_path, rollout_backend="hf", vllm_base_url="", data_designer="on")
    with patch(
        "seiso.rl_verify.data_designer_gen.data_designer_available",
        return_value=False,
    ):
        assert should_use_data_designer(cfg, world_size=1) is True


def test_multigpu_helper_still_detects_tp(tmp_path: Path):
    cfg = _cfg(tmp_path, data_designer="auto", vllm_tensor_parallel=2)
    assert is_multigpu_vllm_run(cfg, world_size=1) is True


def test_records_to_slime_rows_maps_structured_item():
    rows = records_to_slime_rows(
        [
            {
                "stream": "numeric",
                "difficulty": "easy",
                "item": {"problem": "What is 2+2?", "answer": "4"},
            },
            {
                "stream": "choice",
                "difficulty": "medium",
                "item": '{"problem": "Pick A or B. A) 1 B) 2", "answer": "B"}',
            },
            {"stream": "numeric", "item": {"problem": "", "answer": "x"}},  # drop
        ],
        require_thinking_trace=True,
        thinking_instruction="Show work in <think>...</think>.",
    )
    assert len(rows) == 2
    assert rows[0]["label"] == "4"
    assert rows[0]["answer"] == "4"
    assert rows[0]["reward"] == "numeric"
    content = rows[0]["prompt"][0]["content"]
    assert "2+2" in content
    # Thinking instruction is applied at generation time, not baked into rows.
    assert "<think>" not in content
    assert rows[0]["metadata"]["generator"] == "nvidia.nemo.data_designer"
    assert rows[1]["reward"] == "choice"
    assert rows[1]["label"] == "B"


def test_materialize_for_slime_config_uses_designer(tmp_path: Path):
    from seiso.rl_verify import data_designer_gen as ddg
    from seiso.rl_verify.data_gen import DataGenResult

    cfg = _cfg(tmp_path, data_gen=True, data_gen_count=4, output_dir=tmp_path / "out")
    out = tmp_path / "out" / "slime_generated.jsonl"
    fake = DataGenResult(
        rows=[
            {
                "prompt": [{"role": "user", "content": "q"}],
                "label": "1",
                "answer": "1",
                "metadata": {"rm_type": "numeric"},
                "reward": "numeric",
            }
        ],
        stream_counts={"numeric": 1},
        difficulty_counts={"easy": 1},
        seed=17,
    )

    with patch.object(ddg, "materialize_data_designer_corpus", return_value=fake) as mocked:
        result = ddg.materialize_for_slime_config(cfg, out_path=out, count=4, world_size=2)
    assert result is fake
    assert mocked.called
    call_cfg = mocked.call_args[0][1]
    assert "8000" in call_cfg.vllm_base_url
    assert call_cfg.count == 4


def test_trainer_uses_data_designer_when_available(tmp_path: Path):
    from seiso.rl_verify.data_gen import DataGenResult
    from seiso.slime.trainer import (
        _DistributedSlimeContext,
        _maybe_materialize_data_gen,
    )

    cfg = _cfg(
        tmp_path,
        data_gen=True,
        data_gen_count=3,
        data_gen_source="data_designer",
        data_designer="on",
        data_gen_filename="gen.jsonl",
        output_dir=tmp_path / "run",
    )
    dist = _DistributedSlimeContext(
        enabled=True,
        world_size=2,
        rank=0,
        local_rank=0,
        device="cpu",
    )
    fake = DataGenResult(
        rows=[{"prompt": "x", "label": "1", "answer": "1", "metadata": {}, "reward": "numeric"}],
        stream_counts={"numeric": 1},
        difficulty_counts={},
        seed=0,
    )

    written = tmp_path / "run" / "gen.jsonl"

    def _materialize(config, *, out_path, count, world_size=1):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('{"prompt":"x","label":"1"}\n', encoding="utf-8")
        return fake

    with (
        patch(
            "seiso.rl_verify.data_designer_gen.should_use_data_designer",
            return_value=True,
        ),
        patch(
            "seiso.rl_verify.data_designer_gen.data_designer_available",
            return_value=True,
        ),
        patch(
            "seiso.rl_verify.data_designer_gen.materialize_for_slime_config",
            side_effect=_materialize,
        ) as m_dd,
        patch(
            "seiso.slime.trainer._distributed_barrier",
        ),
    ):
        out_cfg = _maybe_materialize_data_gen(cfg, dist)

    assert m_dd.called
    assert out_cfg.dataset == written
    assert written.is_file()
    summary = (tmp_path / "run" / "slime_data_gen_summary.json").read_text(encoding="utf-8")
    assert "nvidia.nemo.data_designer" in summary


def test_trainer_fails_without_data_designer(tmp_path: Path):
    from seiso.slime.trainer import (
        _DistributedSlimeContext,
        _maybe_materialize_data_gen,
    )

    cfg = _cfg(
        tmp_path,
        rollout_backend="hf",
        vllm_base_url="",
        data_gen=True,
        data_gen_count=2,
        data_gen_source="auto",
        data_gen_filename="gen.jsonl",
        output_dir=tmp_path / "run_hf",
    )
    dist = _DistributedSlimeContext(
        enabled=False,
        world_size=1,
        rank=0,
        local_rank=0,
        device="cpu",
    )

    with (
        patch(
            "seiso.rl_verify.data_designer_gen.should_use_data_designer",
            return_value=False,
        ),
        patch(
            "seiso.rl_verify.data_designer_gen.data_designer_available",
            return_value=False,
        ),
        patch("seiso.slime.trainer._distributed_barrier"),
        pytest.raises(
            RuntimeError, match="data_gen_source|Data Designer|materialize source"
        ),
    ):
        _maybe_materialize_data_gen(cfg, dist)


def test_example_vllm_config_defaults_hub_materialize():
    from seiso.training.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/example_training_slime_vllm.yaml")
    slime = cfg.to_single_gpu_slime_config()
    assert slime.data_gen is True
    assert slime.data_gen_source == "dataset"
    assert slime.dataset_ref == "open-r1/OpenR1-Math-220k"
    assert slime.data_designer == "off"
    assert slime.eval_dataset is None  # materialize auto-splits held-out
    assert slime.require_held_out_eval is True
    slime.validate()
