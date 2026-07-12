"""Regressions for GenerationPlan, profiles, dFlash tokens, torch KV stream."""

from __future__ import annotations

import pytest


def test_generation_plan_roundtrip():
    from seiso.inference.plan import GenerationPlan, generation_plan_from_updates

    plan = GenerationPlan(
        model_path="/tmp/m.gguf",
        inference_backend="llamacpp",
        max_tokens=256,
        n_ctx=4096,
        model_format="gguf",
    )
    updates = plan.to_payload_updates()
    assert updates["model_path"] == "/tmp/m.gguf"
    assert updates["n_ctx"] == 4096
    rebuilt = generation_plan_from_updates(updates)
    assert rebuilt is not None
    assert rebuilt.max_tokens == 256
    assert rebuilt.inference_backend == "llamacpp"


def test_generation_plan_skips_router():
    from seiso.inference.plan import generation_plan_from_updates

    assert (
        generation_plan_from_updates(
            {
                "use_model_router": True,
                "model_path": None,
                "inference_backend": "router",
            }
        )
        is None
    )


def test_inference_profile_seeds_without_clobber(monkeypatch):
    from seiso.inference import profiles

    monkeypatch.setenv("SEISO_STREAM_BATCH_CHARS", "99")
    monkeypatch.delenv("SEISO_SIDECAR_PERF_MODE", raising=False)
    resolved = profiles.apply_inference_profile("throughput")
    assert resolved == "throughput"
    import os

    assert os.environ["SEISO_STREAM_BATCH_CHARS"] == "99"
    assert os.environ.get("SEISO_SIDECAR_PERF_MODE") == "1"


def test_resolve_inference_profile_fallback(monkeypatch):
    from seiso.inference.profiles import resolve_inference_profile

    monkeypatch.setenv("SEISO_INFERENCE_PROFILE", "nope")
    assert resolve_inference_profile() == "interactive"


def test_pool_pinned_n_ctx_reused_by_llamaswap_payload(monkeypatch, tmp_path):
    from seiso.inference.model_pool import ModelPool
    from seiso.inference.runner import LocalInferenceRunner

    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    pool = ModelPool()
    pool._active = type(
        "A",
        (),
        {
            "key": f"llamaswap:{model.resolve()}",
            "backend": type("B", (), {"value": "llamaswap"})(),
            "handle": object(),
            "meta": {
                "path": str(model.resolve()),
                "norm_path": str(model.resolve()),
                "n_ctx": 8192,
                "sidecar": True,
            },
        },
    )()
    runner = LocalInferenceRunner()
    runner._pool = pool
    out = runner._llamaswap_payload(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128},
        str(model.resolve()),
    )
    assert out["sidecar_num_ctx"] == 8192


def test_plan_sidecar_reuses_pinned_ctx(monkeypatch):
    from seiso.inference.llamaswap import plan_sidecar_request

    monkeypatch.setattr(
        "seiso.inference.llamaswap._sidecar_context_ceiling",
        lambda _payload, _path: 8192,
    )
    monkeypatch.setattr(
        "seiso.inference.llamaswap.sidecar_vram_context_cap",
        lambda _path, ceiling, **_kw: ceiling,
    )
    monkeypatch.setattr(
        "seiso.inference.llamaswap._sidecar_native_max_tokens",
        lambda n: n,
    )
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 128,
        "sidecar_num_ctx": 4096,
    }
    _messages, num_ctx, max_tokens = plan_sidecar_request(payload, "/tmp/m.gguf")
    assert num_ctx == 4096
    assert max_tokens == 128


def test_sidecar_keep_alive_active_uses_profile(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_KEEP_ALIVE", raising=False)
    monkeypatch.setenv("SEISO_INFERENCE_PROFILE", "interactive")
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_perf_mode", lambda: False)
    monkeypatch.setattr(llamaswap, "_sidecar_headroom_mb", lambda: 20_000)
    assert llamaswap.sidecar_ollama_keep_alive(active=True) == "15m"
    assert llamaswap.sidecar_ollama_keep_alive(active=False) == "2m"


def test_dflash_draft_infer_prefers_token_path():
    from seiso.inference.model_pool import DflashDraftHandle, dflash_draft_infer

    calls: list[dict] = []

    class _FakeDraftLlm:
        def tokenize(self, data, add_bos=False):
            text = data.decode("utf-8")
            return [ord(c) % 97 for c in text]

        def __call__(self, prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            return {"choices": [{"text": " next"}]}

    handle = DflashDraftHandle(_FakeDraftLlm())
    assert dflash_draft_infer(handle, "ab", max_tokens=2) == " next"
    assert dflash_draft_infer(handle, "abc", max_tokens=2) == " next"
    assert isinstance(calls[0]["prompt"], list)
    assert isinstance(calls[1]["prompt"], list)
    assert calls[1].get("cache_prompt") is True
    assert handle._last_tokens


def test_llama_prompt_lookup_opt_in(monkeypatch):
    from seiso.inference.model_pool import llama_load

    monkeypatch.setenv("SEISO_LLAMA_PROMPT_LOOKUP", "0")
    assert llama_load._llama_prompt_lookup_draft() is None

    monkeypatch.setenv("SEISO_LLAMA_PROMPT_LOOKUP", "1")

    class _Draft:
        pass

    class _Mod:
        LlamaPromptLookupDecoding = staticmethod(lambda num_pred_tokens=8: _Draft())

    import sys

    monkeypatch.setitem(sys.modules, "llama_cpp.llama_speculative", _Mod())
    draft = llama_load._llama_prompt_lookup_draft()
    assert isinstance(draft, _Draft)


def test_torch_kv_stream_greedy():
    import torch

    from seiso.inference.torch_stream import iter_torch_kv_tokens

    class _Tok:
        eos_token_id = 99
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(x)) for x in ids)

    class _Out:
        def __init__(self, logits, past):
            self.logits = logits
            self.past_key_values = past

    class _Model:
        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, **kwargs):
            self.calls += 1
            vocab = torch.zeros(1, 1, 100)
            # Prefill + first decode → 4; second decode → 5; then eos.
            if self.calls <= 2:
                vocab[0, 0, 4] = 10.0
            elif self.calls == 3:
                vocab[0, 0, 5] = 10.0
            else:
                vocab[0, 0, 99] = 10.0
            return _Out(vocab, past=object())

    chunks = list(
        iter_torch_kv_tokens(
            model=_Model(),
            tokenizer=_Tok(),
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=4,
            temperature=0.0,
        )
    )
    text = "".join(c.text for c in chunks)
    assert "4" in text
    assert "5" in text


def test_torch_kv_stream_stops_when_pad_is_eos():
    import torch

    from seiso.inference.torch_stream import iter_torch_kv_tokens

    class _Tok:
        eos_token_id = 0
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(x)) for x in ids)

    class _Model:
        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, **kwargs):
            self.calls += 1
            logits = torch.zeros(1, input_ids.shape[1], 4)
            logits[..., 0] = 10.0
            return type("Out", (), {"logits": logits, "past_key_values": object()})()

    model = _Model()
    chunks = list(
        iter_torch_kv_tokens(
            model=model,
            tokenizer=_Tok(),
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=8,
            temperature=0.0,
        )
    )

    # Immediate EOS (pad == eos) yields no text, only a stop marker for auto-continue.
    assert all(not part.text for part in chunks)
    assert chunks and chunks[-1].finish_reason == "stop"
    assert model.calls == 1


def test_speculative_sampling_falls_back_to_target_only(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner
    from seiso.inference.streaming import StreamToken

    runner = LocalInferenceRunner()
    calls: list[str] = []

    def _target_only(_payload, _model_path, _should_stop):
        calls.append("target")
        yield StreamToken("sampled")

    monkeypatch.setattr(runner, "_torch_stream", _target_only)
    monkeypatch.setattr(
        runner._pool,
        "get_torch_speculative",
        lambda *_args, **_kwargs: pytest.fail("speculative pair should not load"),
    )

    chunks = list(
        runner._torch_speculative_stream(
            {
                "draft_model_path": "/tmp/draft",
                "temperature": 0.7,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "/tmp/target",
            lambda: False,
        )
    )

    assert calls == ["target"]
    assert "".join(chunk.text for chunk in chunks) == "sampled"


def test_speculative_low_memory_falls_back_to_target_only(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner
    from seiso.inference.streaming import StreamToken

    runner = LocalInferenceRunner()
    monkeypatch.setattr(
        runner._pool, "torch_speculative_pair_fits", lambda *_args: False
    )
    monkeypatch.setattr(
        runner,
        "_torch_stream",
        lambda *_args: iter([StreamToken("fallback")]),
    )
    monkeypatch.setattr(
        runner._pool,
        "get_torch_speculative",
        lambda *_args, **_kwargs: pytest.fail("speculative pair should not load"),
    )

    chunks = list(
        runner._torch_speculative_stream(
            {
                "draft_model_path": "/tmp/draft",
                "temperature": 0,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "/tmp/target",
            lambda: False,
        )
    )

    assert "".join(chunk.text for chunk in chunks) == "fallback"


@pytest.mark.asyncio
async def test_prepare_attaches_generation_plan(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from forge.services import inference_chat
    from seiso.inference.backends import BACKEND_LLAMACPP

    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")

    async def option(*_a, **_k):
        return {
            "id": "m1",
            "name": "Model",
            "path": str(model),
            "format": "gguf",
            "selectable": True,
            "default_backend": BACKEND_LLAMACPP,
            "backends": [BACKEND_LLAMACPP],
            "size_bytes": 10,
        }

    monkeypatch.setattr(inference_chat, "get_inference_option", option)
    monkeypatch.setattr(
        inference_chat, "assert_model_fits_for_load", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        inference_chat,
        "assert_backend_runtime_available",
        lambda *_a, **_k: None,
    )

    updates = await inference_chat.prepare_local_chat_target(
        object(),
        "u1",
        SimpleNamespace(data_dir=tmp_path),
        model_id="m1",
        inference_backend="auto",
        check_memory=False,
        sanitize=False,
    )
    assert "generation_plan" in updates
    assert updates["generation_plan"]["model_path"] == str(model)
