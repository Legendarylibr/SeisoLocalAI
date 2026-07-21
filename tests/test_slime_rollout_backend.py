"""Tests for slime online rollout backends (hf / sglang / vllm)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.rollout_backend import (
    SGLangRolloutClient,
    VLLMRolloutClient,
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
        require_held_out_eval=False,
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


def test_auto_prefers_vllm_over_sglang_when_both_set(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="auto",
        sglang_base_url="http://127.0.0.1:30000",
        vllm_base_url="http://127.0.0.1:8000",
    )
    assert resolve_rollout_backend(cfg, world_size=1) == "hf"
    assert resolve_rollout_backend(cfg, world_size=2) == "vllm"


def test_sglang_requires_base_url(tmp_path: Path):
    cfg = _cfg(tmp_path, rollout_backend="sglang", sglang_base_url="")
    with pytest.raises(ValueError, match="sglang_base_url"):
        validate_rollout_backend_config(cfg)


def test_vllm_requires_base_url(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SEISO_MANAGED_VLLM_PORT", raising=False)
    with patch(
        "seiso.slime.rollout_backend.resolve_vllm_base_url",
        return_value="",
    ):
        cfg = _cfg(tmp_path, rollout_backend="vllm", vllm_base_url="")
        with pytest.raises(ValueError, match="vllm_base_url"):
            validate_rollout_backend_config(cfg)


def test_vllm_accepts_base_url(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="vllm",
        vllm_base_url="http://127.0.0.1:8000",
    )
    cfg.validate()
    assert resolve_rollout_backend(cfg, world_size=2) == "vllm"


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
        "seiso.slime.rollout_http.urllib.request.urlopen",
        return_value=_Resp(),
    ):
        assert client.complete("prompt") == " 42 "


def test_sglang_client_complete_with_tokens_reads_output_ids(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="sglang",
        sglang_base_url="http://127.0.0.1:30000",
        sglang_model="served-model",
    )
    client = SGLangRolloutClient.from_config(cfg)
    payload = {
        "choices": [
            {
                "text": "answer",
                "meta_info": {"output_token_ids": [7, 8, 9]},
            }
        ]
    }

    class _Resp:
        def read(self):
            return json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch(
        "seiso.slime.rollout_http.urllib.request.urlopen",
        return_value=_Resp(),
    ):
        text, tids = client.complete_with_tokens("prompt")
    assert text == "answer"
    assert tids == [7, 8, 9]


def test_build_sequence_tensors_prefers_server_token_ids(tmp_path: Path):
    import torch

    from seiso.slime.rollout_backend import build_sequence_tensors

    class _Tok:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            # Distinct encoding so server ids are observable if used.
            ids = [1, 1] if text == "P" else [99, 99, 99]
            return {"input_ids": torch.tensor([ids])}

    cfg = _cfg(tmp_path, max_prompt_tokens=16, max_new_tokens=8)
    rows = build_sequence_tensors(
        tokenizer=_Tok(),
        prompts=["P"],
        completions=["ignored"],
        config=cfg,
        torch=torch,
        device="cpu",
        completion_token_ids=[[10, 11, 2]],
    )
    assert rows[0]["input_ids"].tolist() == [1, 1, 10, 11, 2]
    # Keep EOS when pad != eos; all response tokens active.
    assert rows[0]["response_mask"].tolist() == [False, False, True, True, True]


def test_sglang_update_weights_from_disk(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="sglang",
        sglang_base_url="http://127.0.0.1:30000",
    )
    client = SGLangRolloutClient.from_config(cfg)
    seen: dict[str, object] = {}

    class _Resp:
        def read(self):
            return json.dumps({"success": True}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = req.data
        return _Resp()

    with patch(
        "seiso.slime.rollout_http.urllib.request.urlopen",
        side_effect=_urlopen,
    ):
        out = client.update_weights_from_disk("/tmp/weights", weight_version="v3")
    assert out["success"] is True
    assert seen["url"] == "http://127.0.0.1:30000/update_weights_from_disk"
    assert seen["method"] == "POST"
    body = json.loads(seen["body"].decode("utf-8"))
    assert body["model_path"] == "/tmp/weights"
    assert body["weight_version"] == "v3"


def test_sync_sglang_weights_noop_for_hf_backend(tmp_path: Path):
    from seiso.slime.rollout_backend import sync_sglang_weights_from_actor

    cfg = _cfg(tmp_path, rollout_backend="hf", output_dir=tmp_path / "out")
    path = sync_sglang_weights_from_actor(
        model=object(),
        tokenizer=object(),
        config=cfg,
        step=1,
        is_main=True,
    )
    assert path is None


def test_sync_vllm_weights_noop_for_hf_backend(tmp_path: Path):
    from seiso.slime.rollout_backend import sync_vllm_weights_from_actor

    cfg = _cfg(tmp_path, rollout_backend="hf", output_dir=tmp_path / "out")
    path = sync_vllm_weights_from_actor(
        model=object(),
        tokenizer=object(),
        config=cfg,
        step=1,
        is_main=True,
    )
    assert path is None


def test_vllm_client_strips_v1_suffix_and_loads_lora(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="vllm",
        vllm_base_url="http://127.0.0.1:8000/v1",
        vllm_model="base-model",
        vllm_lora_name="policy_lora",
        use_lora=True,
    )
    client = VLLMRolloutClient.from_config(cfg)
    assert client.base_url == "http://127.0.0.1:8000"
    assert client._active_model == "policy_lora"

    seen: list[dict[str, object]] = []

    class _Resp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _urlopen(req, timeout=0):
        body = req.data.decode("utf-8") if req.data else ""
        seen.append({"url": req.full_url, "method": req.get_method(), "body": body})
        if "unload_lora_adapter" in req.full_url:
            return _Resp(b'{"ok": true}')
        if "load_lora_adapter" in req.full_url:
            return _Resp(b"Success: LoRA adapter 'policy_lora' added successfully")
        return _Resp(b'{"choices":[{"text":"ok"}]}')

    with patch(
        "seiso.slime.rollout_http.urllib.request.urlopen",
        side_effect=_urlopen,
    ):
        client.load_lora_adapter("/tmp/adapter", lora_name="policy_lora")
        assert client.complete("hi") == "ok"

    urls = [s["url"] for s in seen]
    assert any(u.endswith("/v1/load_lora_adapter") for u in urls)
    assert any(u.endswith("/v1/completions") for u in urls)
    load_bodies = [
        json.loads(s["body"])
        for s in seen
        if s["url"].endswith("/v1/load_lora_adapter")
    ]
    assert load_bodies[0]["lora_name"] == "policy_lora"
    assert load_bodies[0]["lora_path"] == "/tmp/adapter"


def test_sglang_engine_urls_dedupes_comma_and_list(tmp_path: Path):
    from seiso.slime.rollout_backend import sglang_engine_urls

    cfg = _cfg(
        tmp_path,
        rollout_backend="sglang",
        sglang_base_url="http://127.0.0.1:30000,http://127.0.0.1:30001/",
        sglang_engine_urls=["http://127.0.0.1:30001", "http://127.0.0.1:30002"],
    )
    urls = sglang_engine_urls(cfg)
    assert urls == [
        "http://127.0.0.1:30000",
        "http://127.0.0.1:30001",
        "http://127.0.0.1:30002",
    ]


def test_prune_weight_versions_keeps_last_n(tmp_path: Path):
    from seiso.slime.rollout_backend import _prune_weight_versions

    root = tmp_path / "weights"
    for name in (
        "weight_v000001",
        "weight_v000002",
        "weight_v000003",
        "delta_v000001",
        "delta_v000002",
        "delta_v000003",
    ):
        d = root / name
        d.mkdir(parents=True)
        (d / "x.txt").write_text("ok", encoding="utf-8")
    _prune_weight_versions(root, keep=2)
    left = sorted(p.name for p in root.iterdir())
    assert left == [
        "delta_v000002",
        "delta_v000003",
        "weight_v000002",
        "weight_v000003",
    ]


def test_from_config_uses_first_engine_url(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="sglang",
        sglang_base_url="http://127.0.0.1:30000,http://127.0.0.1:30001",
    )
    client = SGLangRolloutClient.from_config(cfg)
    assert client.base_url == "http://127.0.0.1:30000"


def test_sglang_url_rejects_non_http(tmp_path: Path):
    cfg = _cfg(
        tmp_path,
        rollout_backend="sglang",
        sglang_base_url="file:///etc/passwd",
    )
    with pytest.raises(ValueError, match="http"):
        SGLangRolloutClient.from_config(cfg)


def test_example_ddp_config_requests_sglang():
    from seiso.training.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/example_training_slime_ddp.yaml")
    slime = cfg.to_single_gpu_slime_config()
    assert slime.rollout_backend == "sglang"
    assert slime.sglang_base_url.startswith("http")
    slime.validate()


def test_example_vllm_ddp_config_requests_vllm():
    from seiso.training.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/example_training_slime_vllm.yaml")
    slime = cfg.to_single_gpu_slime_config()
    assert slime.rollout_backend == "vllm"
    assert slime.vllm_base_url.startswith("http")
    assert slime.use_lora is True
    slime.validate()


def test_example_single_gpu_keeps_hf_backend():
    cfg = SingleGpuSlimeConfig.from_yaml(Path("configs/example_slime_single_gpu.yaml"))
    assert cfg.rollout_backend == "hf"
    cfg.validate()


def test_vllm_engine_urls_strip_v1_and_dedupe(tmp_path: Path):
    from seiso.slime.rollout_backend import vllm_engine_urls

    cfg = _cfg(
        tmp_path,
        rollout_backend="vllm",
        vllm_base_url="http://127.0.0.1:8000/v1,http://127.0.0.1:8001",
        vllm_engine_urls=["http://127.0.0.1:8001/v1", "http://127.0.0.1:8002"],
    )
    urls = vllm_engine_urls(cfg)
    assert urls == [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8002",
    ]
