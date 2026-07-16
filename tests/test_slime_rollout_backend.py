"""Tests for slime online rollout backends (data_gen / sglang)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from seiso.slime_single_gpu.config import SingleGpuSlimeConfig
from seiso.slime_single_gpu.rollout_backend import (
    SGLangRolloutClient,
    format_generation_prompt,
    resolve_rollout_backend,
    validate_rollout_backend_config,
)


def _cfg(tmp_path: Path, **kwargs) -> SingleGpuSlimeConfig:
    base = dict(
        model_id="test/model",
        dataset=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        rollouts_per_prompt=2,
        rollout_batch_size=2,
    )
    base.update(kwargs)
    return SingleGpuSlimeConfig(**base)


def test_default_backend_is_hf(tmp_path: Path):
    cfg = _cfg(tmp_path)
    assert resolve_rollout_backend(cfg, world_size=1) == "hf"
    assert resolve_rollout_backend(cfg, world_size=4) == "hf"
    cfg.validate()


def test_data_gen_alias_maps_to_hf(tmp_path: Path):
    cfg = _cfg(tmp_path, rollout_backend="data_gen")
    assert resolve_rollout_backend(cfg, world_size=1) == "hf"


def test_auto_uses_sglang_only_when_url_and_multi_process(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="auto",
        sglang_base_url="http://127.0.0.1:30000",
    )
    assert resolve_rollout_backend(cfg, world_size=1) == "hf"
    assert resolve_rollout_backend(cfg, world_size=2) == "sglang"


def test_sglang_requires_base_url(tmp_path: Path):
    cfg = _cfg(tmp_path, rollout_backend="sglang", sglang_base_url="")
    with pytest.raises(ValueError, match="sglang_base_url"):
        validate_rollout_backend_config(cfg)


def test_format_generation_prompt_thinking_open(tmp_path: Path):
    cfg = _cfg(tmp_path, require_thinking_trace=True, apply_chat_template=False)

    class _Tok:
        pass

    text = format_generation_prompt(_Tok(), "What is 2+2?", cfg)
    assert text.endswith("<think>")
    assert "What is 2+2?" in text


def test_sglang_client_complete_parses_text(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="sglang",
        sglang_base_url="http://127.0.0.1:30000",
        sglang_model="served-model",
    )
    client = SGLangRolloutClient.from_config(cfg)
    payload = {"choices": [{"text": " 42 "}]}

    class _Resp:
        def read(self):
            return json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch(
        "seiso.slime_single_gpu.rollout_backend.urllib.request.urlopen",
        return_value=_Resp(),
    ):
        assert client.complete("prompt") == " 42 "


def test_example_ddp_config_requests_sglang():
    from seiso.training.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/example_training_slime_ddp.yaml")
    slime = cfg.to_single_gpu_slime_config()
    assert slime.rollout_backend == "sglang"
    assert slime.sglang_base_url.startswith("http")
    slime.validate()


def test_example_single_gpu_keeps_hf_backend():
    cfg = SingleGpuSlimeConfig.from_yaml(Path("configs/example_slime_single_gpu.yaml"))
    assert cfg.rollout_backend == "hf"
    cfg.validate()
