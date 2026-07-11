"""Tests for inference backend selection and model options."""

from __future__ import annotations

import json
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gguf_fixtures import write_arch_gguf as _write_arch_gguf
from gguf_fixtures import write_gguf_u32_metadata as _write_gguf_with_u32_metadata

from seiso.inference.backends import (
    BACKEND_LLAMACPP,
    BACKEND_LLAMASWAP,
    BACKEND_MLX,
    BACKEND_TORCH,
    available_backends,
    clear_gguf_caches,
    gguf_architecture,
    gguf_block_count,
    gguf_context_length,
    gguf_sliding_window,
    gguf_uses_sliding_window_attention,
    prepare_model_path,
    recommend_backend,
    resolve_gguf_file,
    resolve_local_backend,
)


@pytest.fixture(autouse=True)
def _reset_inference_caches():
    from forge.services import inference_models
    from forge.services.hf_connectivity import check_inference_runtime

    inference_models.invalidate_inference_options_cache()
    check_inference_runtime.cache_clear()
    clear_gguf_caches()
    yield
    inference_models.invalidate_inference_options_cache()
    check_inference_runtime.cache_clear()
    clear_gguf_caches()


def test_gguf_recommends_llamacpp(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: False)
    assert recommend_backend(model_path=str(gguf), model_format="gguf") == BACKEND_LLAMACPP


def test_gguf_auto_prefers_llamaswap_on_native_linux_when_enabled(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: True)
    monkeypatch.setattr(
        "seiso.inference.llamaswap.llamaswap_status",
        lambda: SimpleNamespace(available=True),
    )

    assert recommend_backend(model_path=str(gguf), model_format="gguf") == BACKEND_LLAMASWAP
    assert (
        resolve_local_backend(
            model_path=str(gguf),
            model_format="gguf",
            requested="auto",
        )
        == BACKEND_LLAMASWAP
    )


def test_gguf_auto_requires_healthy_llamaswap_on_native_linux(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: True)
    monkeypatch.setattr(
        "seiso.inference.llamaswap.llamaswap_status",
        lambda: SimpleNamespace(available=False, reason="sidecar down"),
    )

    assert recommend_backend(model_path=str(gguf), model_format="gguf") == BACKEND_LLAMASWAP
    assert available_backends(model_path=str(gguf), model_format="gguf") == [BACKEND_LLAMASWAP]
    with pytest.raises(RuntimeError, match="requires an isolated backend"):
        resolve_local_backend(
            model_path=str(gguf),
            model_format="gguf",
            requested="auto",
        )


def test_gguf_explicit_llamacpp_requires_unsafe_native_linux_override(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: True)
    monkeypatch.setattr(
        "seiso.inference.llamaswap.llamaswap_status",
        lambda: SimpleNamespace(available=False, reason="sidecar down"),
    )

    with pytest.raises(RuntimeError, match="requested backend was llamacpp"):
        resolve_local_backend(
            model_path=str(gguf),
            model_format="gguf",
            requested="llamacpp",
        )

    monkeypatch.setenv("SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX", "1")
    assert (
        resolve_local_backend(
            model_path=str(gguf),
            model_format="gguf",
            requested="llamacpp",
        )
        == BACKEND_LLAMACPP
    )


def test_recommend_backend_honors_unsafe_native_linux_override(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: True)
    monkeypatch.setenv("SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX", "1")

    assert recommend_backend(model_path=str(gguf), model_format="gguf") == BACKEND_LLAMACPP


def test_gguf_explicit_llamacpp_uses_healthy_sidecar_on_native_linux(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: True)
    monkeypatch.setattr(
        "seiso.inference.llamaswap.llamaswap_status",
        lambda: SimpleNamespace(available=True),
    )

    with pytest.raises(RuntimeError, match="requested backend was llamacpp"):
        resolve_local_backend(
            model_path=str(gguf),
            model_format="gguf",
            requested="llamacpp",
        )


def _force_bare_metal_linux(monkeypatch, *, nvidia_smi: bool) -> None:
    """Simulate a bare-metal Linux host with/without an nvidia-smi GPU signal."""
    import seiso.inference.backends as backends_mod

    monkeypatch.setattr(backends_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr("seiso.platform.detect_wsl2", lambda: False)
    monkeypatch.setattr("seiso.security.nvidia_boundary.nvidia_smi_visible", lambda: nvidia_smi)
    monkeypatch.delenv("SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX", raising=False)


def test_isolation_required_when_guard_detection_raises(monkeypatch):
    """A torch CUDA probe error must not re-enable the in-process CUDA path."""
    from seiso.inference.backends import _native_linux_requires_isolated_gguf

    def _boom() -> bool:
        raise RuntimeError("CUDA driver/runtime version mismatch")

    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", _boom)
    _force_bare_metal_linux(monkeypatch, nvidia_smi=True)

    assert _native_linux_requires_isolated_gguf() is True


def test_isolation_required_when_only_nvidia_smi_detects_gpu(monkeypatch):
    """Profile-based detection missed the GPU; nvidia-smi fallback still isolates."""
    from seiso.inference.backends import _native_linux_requires_isolated_gguf

    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: False)
    _force_bare_metal_linux(monkeypatch, nvidia_smi=True)

    assert _native_linux_requires_isolated_gguf() is True


def test_no_isolation_on_cpu_only_linux(monkeypatch):
    """CPU-only Linux (no nvidia-smi) keeps in-process llama.cpp available."""
    from seiso.inference.backends import _native_linux_requires_isolated_gguf

    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: False)
    _force_bare_metal_linux(monkeypatch, nvidia_smi=False)

    assert _native_linux_requires_isolated_gguf() is False


def test_resolve_local_backend_never_inprocess_gguf_via_nvidia_smi(monkeypatch, tmp_path: Path):
    """GGUF chat on a Linux+NVIDIA host raises instead of falling to CUDA."""
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")

    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: False)
    _force_bare_metal_linux(monkeypatch, nvidia_smi=True)
    monkeypatch.setattr(
        "seiso.inference.llamaswap.llamaswap_status",
        lambda: SimpleNamespace(available=False, reason="sidecar down"),
    )

    # Recommended backend is the sidecar, never in-process llama.cpp.
    assert recommend_backend(model_path=str(gguf), model_format="gguf") == BACKEND_LLAMASWAP
    # Dispatch refuses to run in-process on CUDA when the sidecar is down.
    with pytest.raises(RuntimeError, match="requires an isolated backend"):
        resolve_local_backend(
            model_path=str(gguf),
            model_format="gguf",
            requested="auto",
        )


def test_probe_torch_gpus_swallows_cuda_runtime_error(monkeypatch):
    """A broken CUDA runtime yields an empty probe, not a propagated exception."""
    import sys
    import types

    from seiso.hardware.probes.torch_cuda import probe_torch_gpus

    fake_torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("CUDA unknown error")

    fake_torch.cuda = _Cuda()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert probe_torch_gpus() == []


def test_torch_prompt_trim_drops_old_turns_and_clamps_generation():
    from seiso.inference.runner import _trim_torch_messages_to_context

    class FakeTokenizer:
        model_max_length = 48

        def __call__(self, prompt: str, **_kwargs):
            return {"input_ids": prompt.split()}

    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=48))
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "old " * 200},
        {"role": "assistant", "content": "history " * 200},
        {"role": "user", "content": "answer this now"},
    ]

    budget = _trim_torch_messages_to_context(
        messages,
        model=model,
        tokenizer=tokenizer,
        max_tokens=128,
    )

    assert budget.messages[-1]["content"] == "answer this now"
    assert all("history" not in message["content"] for message in budget.messages)
    assert budget.max_tokens < 128
    assert budget.input_tokens + budget.max_tokens <= budget.context_limit


def test_torch_prepare_inputs_returns_clamped_generation_budget():
    from seiso.inference.runner import LocalInferenceRunner

    class FakeTensor:
        shape = (1, 4)

        def to(self, *_args, **_kwargs):
            return self

    class FakeTokenizer:
        model_max_length = 16

        def __call__(self, prompt: str, **kwargs):
            if kwargs.get("return_tensors"):
                return {"input_ids": FakeTensor()}
            return {"input_ids": prompt.split()}

    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=16),
        device=SimpleNamespace(type="cpu"),
    )

    _inputs, _input_len, max_tokens = LocalInferenceRunner._torch_prepare_inputs(
        model,
        [{"role": "user", "content": "hello"}],
        FakeTokenizer(),
        max_tokens=128,
    )

    assert max_tokens < 128


def test_torch_prompt_trim_raises_when_prompt_cannot_fit_context():
    from seiso.inference.runner import _trim_torch_messages_to_context

    class FakeTokenizer:
        model_max_length = 4

        def __call__(self, prompt: str, **_kwargs):
            return {"input_ids": prompt.split()}

    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4))

    with pytest.raises(RuntimeError, match="exceeds model context"):
        _trim_torch_messages_to_context(
            [{"role": "user", "content": ["untrimmable", "payload"]}],
            model=model,
            tokenizer=FakeTokenizer(),
            max_tokens=8,
        )


def test_torch_oom_retry_does_not_increase_clamped_generation(monkeypatch):
    from seiso.inference import runner

    seen: list[int] = []

    def fake_generate(_model, gen_kwargs):
        seen.append(gen_kwargs["max_new_tokens"])
        if len(seen) == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    monkeypatch.setattr(runner, "generate_with_cache_fallback", fake_generate)
    monkeypatch.setattr(runner, "release_cached_memory", lambda sync=False: None)

    result = runner._torch_generate_with_oom_retry(None, {"max_new_tokens": 5})

    assert result == "ok"
    assert seen == [5, 2]


def test_torch_streaming_oom_does_not_retry_existing_streamer(monkeypatch):
    from seiso.inference import runner

    calls = 0

    def fake_generate(_model, _gen_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(runner, "generate_with_cache_fallback", fake_generate)

    with pytest.raises(RuntimeError, match="out of memory"):
        runner._torch_generate_with_oom_retry(
            None,
            {"max_new_tokens": 5},
            retry_on_oom=False,
        )

    assert calls == 1


def test_llama_prompt_budget_uses_tokenizer_and_drops_old_turns():
    from seiso.inference.runner import _fit_llama_messages_to_context

    class FakeLlama:
        def tokenize(self, prompt: bytes, **_kwargs):
            return prompt.decode("utf-8").split()

    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "old " * 200},
        {"role": "assistant", "content": "history " * 200},
        {"role": "user", "content": "answer now"},
    ]

    budget = _fit_llama_messages_to_context(
        FakeLlama(),
        messages,
        n_ctx=64,
        max_tokens=20,
    )

    assert budget.messages[-1]["content"] == "answer now"
    assert all("history" not in str(message["content"]) for message in budget.messages)
    assert budget.input_tokens + budget.max_tokens < budget.context_limit


def test_llama_prompt_budget_raises_before_prefill_when_prompt_cannot_fit():
    from seiso.inference.runner import _fit_llama_messages_to_context

    class FakeLlama:
        def tokenize(self, prompt: bytes, **_kwargs):
            return list(prompt)

    with pytest.raises(RuntimeError, match="llama.cpp prompt exceeds context"):
        _fit_llama_messages_to_context(
            FakeLlama(),
            [{"role": "user", "content": ["untrimmable"]}],
            n_ctx=8,
            max_tokens=4,
        )


def test_native_linux_llama_context_defaults_to_stable_bucket(monkeypatch):
    from seiso.inference.tuning import estimate_llama_n_ctx

    monkeypatch.delenv("SEISO_LLAMA_DYNAMIC_CTX", raising=False)
    monkeypatch.delenv("SEISO_LLAMA_NATIVE_STABLE_N_CTX", raising=False)
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: True)

    n_ctx = estimate_llama_n_ctx(
        [{"role": "user", "content": "long prompt " * 4000}],
        max_tokens=512,
        default=4096,
    )

    assert n_ctx == 2048


def test_native_linux_llama_context_stable_bucket_requires_sticky_override(monkeypatch):
    from seiso.inference.tuning import estimate_llama_n_ctx

    monkeypatch.delenv("SEISO_LLAMA_DYNAMIC_CTX", raising=False)
    monkeypatch.setenv("SEISO_LLAMA_NATIVE_STABLE_N_CTX", "4096")
    monkeypatch.setenv("SEISO_LLAMA_UNSAFE_STICKY_CTX", "1")
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda: True)

    n_ctx = estimate_llama_n_ctx(
        [{"role": "user", "content": "long prompt " * 4000}],
        max_tokens=512,
        default=4096,
    )

    assert n_ctx == 4096


def _complete_hf_gguf_metadata(filename: str = "model-q4.gguf") -> dict:
    return {
        "repo_id": "org/Model",
        "gguf_repo": "org/Model",
        "gguf_files": [filename],
    }


def test_safetensors_recommends_torch_or_mlx(tmp_path: Path):
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"x")
    backend = recommend_backend(model_path=str(model_dir), model_format="safetensors")
    assert backend in {BACKEND_TORCH, "mlx"}


def test_safetensors_inventory_exposes_torch_and_mlx_fallbacks(monkeypatch, tmp_path: Path):
    from seiso.inference import backends
    from seiso.models.loader import Backend

    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(backends, "detect_backend", lambda: Backend.MLX)
    monkeypatch.setattr(backends.platform, "system", lambda: "Darwin")

    assert available_backends(model_path=str(model_dir), model_format="safetensors") == [
        BACKEND_MLX
    ]


def test_gguf_architecture_reads_metadata(tmp_path: Path):
    gguf = tmp_path / "model.gguf"
    _write_arch_gguf(gguf, "llama")

    assert gguf_architecture(str(gguf)) == "llama"


def test_gguf_metadata_reader_collects_context_blocks_and_swa(tmp_path: Path):
    gguf = tmp_path / "model.gguf"
    _write_gguf_with_u32_metadata(
        gguf,
        [
            (b"llama.context_length", 8192),
            (b"llama.block_count", 32),
            (b"llama.attention.sliding_window", 4096),
        ],
    )

    assert gguf_architecture(str(gguf)) == "llama"
    assert gguf_context_length(str(gguf)) == 8192
    assert gguf_block_count(str(gguf)) == 32
    assert gguf_uses_sliding_window_attention(str(gguf)) is True
    assert gguf_sliding_window(str(gguf)) == 4096


def test_available_backends_allows_dflash_draft_for_speculative(tmp_path: Path):
    gguf = tmp_path / "draft.gguf"
    _write_arch_gguf(gguf, "dflash-draft")

    # dflash-draft is allowed by the local host policy when used as speculative draft model.
    backends = available_backends(model_path=str(gguf), model_format="gguf")
    assert BACKEND_LLAMACPP in backends


def test_is_dflash_draft_requires_gguf_and_name_or_arch(tmp_path: Path):
    from seiso.inference.backends import is_dflash_draft

    dflash = tmp_path / "model-dflash.gguf"
    _write_arch_gguf(dflash, "llama")
    assert is_dflash_draft(str(dflash))

    arch = tmp_path / "draft.gguf"
    _write_arch_gguf(arch, "dflash")
    assert is_dflash_draft(str(arch))

    # Bare "-draft" / draft- prefix without dflash signal is not enough
    plain = tmp_path / "my-draft-model.gguf"
    _write_arch_gguf(plain, "llama")
    assert not is_dflash_draft(str(plain))

    # Non-GGUF paths are never dflash drafts
    assert not is_dflash_draft(str(tmp_path / "draft-model"))


@pytest.mark.asyncio
async def test_resolve_preload_context_uses_chat_sized_context(monkeypatch, tmp_path):
    from forge.services import inference_chat

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    async def fake_prepare(*_args, **kwargs):
        return {
            "model_path": str(model_path),
            "model_format": "gguf",
            "inference_backend": BACKEND_LLAMACPP,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "n_ctx": kwargs.get("n_ctx"),
            "model_name": "Model",
            "size_bytes": 123,
        }

    monkeypatch.setattr(inference_chat, "prepare_local_chat_target", fake_prepare)

    ctx = await inference_chat.resolve_preload_context(
        object(),
        "u1",
        SimpleNamespace(data_dir=tmp_path),
        "m1",
        "auto",
        max_tokens=4096,
        n_ctx=8192,
    )

    assert ctx["payload"]["max_tokens"] == 4096
    assert ctx["payload"]["n_ctx"] == 8192
    assert ctx["payload"]["model_path"] == str(model_path)
    assert ctx["backend"] == BACKEND_LLAMACPP


@pytest.mark.asyncio
async def test_resolve_explicit_model_path_checks_selected_backend(monkeypatch, tmp_path):
    from forge.services import inference_chat

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    seen: dict[str, str | None] = {}

    async def fake_resolve_model_path(*_args, **_kwargs):
        return str(model_path)

    def fake_assert_model_fits(path: str, *, mode: str, backend: str | None = None):
        seen["path"] = path
        seen["mode"] = mode
        seen["backend"] = backend

    monkeypatch.setattr(
        inference_chat,
        "resolve_model_path",
        fake_resolve_model_path,
    )
    monkeypatch.setattr(
        inference_chat,
        "assert_model_fits_for_load",
        fake_assert_model_fits,
    )
    monkeypatch.setattr(
        inference_chat,
        "assert_backend_runtime_available",
        lambda _backend: None,
    )

    updates = await inference_chat.resolve_explicit_model_path(
        object(),
        "u1",
        SimpleNamespace(data_dir=tmp_path),
        model_path=str(model_path),
        inference_backend=BACKEND_LLAMACPP,
    )

    assert updates["model_path"] == str(model_path)
    assert updates["inference_backend"] == BACKEND_LLAMACPP
    assert seen == {
        "path": str(model_path),
        "mode": "chat",
        "backend": BACKEND_LLAMACPP,
    }


@pytest.mark.asyncio
async def test_resolve_explicit_model_path_rejects_unavailable_backend(monkeypatch, tmp_path):
    from fastapi import HTTPException

    from forge.services import inference_chat

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    async def fake_resolve_model_path(*_args, **_kwargs):
        return str(model_path)

    monkeypatch.setattr(
        inference_chat,
        "resolve_model_path",
        fake_resolve_model_path,
    )

    def unavailable(_backend):
        raise HTTPException(400, "Inference backend 'llamacpp' is not available")

    monkeypatch.setattr(
        inference_chat,
        "assert_backend_runtime_available",
        unavailable,
    )

    with pytest.raises(HTTPException) as exc_info:
        await inference_chat.resolve_explicit_model_path(
            object(),
            "u1",
            SimpleNamespace(data_dir=tmp_path),
            model_path=str(model_path),
            inference_backend=BACKEND_LLAMACPP,
        )

    assert exc_info.value.status_code == 400
    assert "not available" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_resolve_explicit_model_path_rejects_incompatible_backend(monkeypatch, tmp_path):
    from fastapi import HTTPException

    from forge.services import inference_chat

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    async def fake_resolve_model_path(*_args, **_kwargs):
        return str(model_path)

    monkeypatch.setattr(
        inference_chat,
        "resolve_model_path",
        fake_resolve_model_path,
    )

    with pytest.raises(HTTPException) as exc_info:
        await inference_chat.resolve_explicit_model_path(
            object(),
            "u1",
            SimpleNamespace(data_dir=tmp_path),
            model_path=str(model_path),
            inference_backend=BACKEND_TORCH,
        )

    assert exc_info.value.status_code == 400
    assert "cannot load GGUF" in str(exc_info.value.detail)


def test_recommend_backend_detects_extensionless_hf_blob(tmp_path: Path):
    blob = tmp_path / "hf_cache" / "models--org--Model-GGUF" / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    _write_arch_gguf(blob, "llama")

    assert recommend_backend(model_path=str(blob)) == BACKEND_LLAMACPP
    assert resolve_gguf_file(str(blob)) == blob.absolute()


def test_prepare_model_path_uses_parent_for_hf_weight_file(tmp_path: Path):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"x")

    assert prepare_model_path(str(weights), BACKEND_TORCH) == str(model_dir.absolute())
    assert prepare_model_path(str(weights), BACKEND_MLX) == str(model_dir.absolute())


def test_prepare_model_path_preserves_standalone_weight_file(tmp_path: Path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"x")

    assert prepare_model_path(str(weights), BACKEND_TORCH) == str(weights)


def test_resolve_gguf_file_picks_largest(tmp_path: Path):
    small = tmp_path / "small.gguf"
    large = tmp_path / "large.gguf"
    small.write_bytes(b"a")
    large.write_bytes(b"a" * 10)
    assert resolve_gguf_file(str(tmp_path)).name == "large.gguf"


def test_resolve_gguf_file_prefers_first_shard_without_sorting_by_size(tmp_path: Path):
    shard_two = tmp_path / "model-00002-of-00002.gguf"
    shard_one = tmp_path / "model-00001-of-00002.gguf"
    shard_two.write_bytes(b"a" * 100)
    shard_one.write_bytes(b"a")

    assert resolve_gguf_file(str(tmp_path)).name == shard_one.name


def test_resolve_gguf_file_preserves_symlink_path(tmp_path: Path):
    blob = tmp_path / "hf_cache" / "models--org--Model-GGUF" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf")
    snapshot = (
        tmp_path / "hf_cache" / "models--org--Model-GGUF" / "snapshots" / "rev" / "model-q4.gguf"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to("../../blobs/abc")

    assert resolve_gguf_file(str(snapshot)) == snapshot.absolute()


def test_model_pool_passes_preserved_path_to_loader(tmp_path: Path):
    from seiso.inference.model_pool import BackendKind, ModelPool

    blob = tmp_path / "hf_cache" / "models--org--Model-GGUF" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf")
    snapshot = (
        tmp_path / "hf_cache" / "models--org--Model-GGUF" / "snapshots" / "rev" / "model-q4.gguf"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to("../../blobs/abc")

    seen: list[str] = []
    pool = ModelPool()

    def loader(path: str) -> object:
        seen.append(path)
        return object()

    pool.switch(str(snapshot), BackendKind.LLAMA, loader)
    pool.unload_all()

    assert seen == [str(snapshot.absolute())]


def test_resolve_local_backend_auto():
    assert (
        resolve_local_backend(
            model_path="/tmp/model.gguf",
            model_format="gguf",
            requested="auto",
        )
        == BACKEND_LLAMACPP
    )


def test_resolve_local_backend_rejects_incompatible_explicit_backend(tmp_path: Path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"x")

    with pytest.raises(ValueError, match="cannot load GGUF"):
        resolve_local_backend(
            model_path=str(gguf),
            model_format="gguf",
            requested=BACKEND_TORCH,
        )

    with pytest.raises(ValueError, match="requires a GGUF"):
        resolve_local_backend(
            model_path=str(model_dir),
            model_format="safetensors",
            requested=BACKEND_LLAMASWAP,
        )


def test_llamaswap_engine_prefers_llamacpp_on_macos(monkeypatch):
    from seiso.inference import llamaswap, sidecar_runtime

    monkeypatch.delenv("SEISO_LLAMASWAP_ENGINE", raising=False)
    monkeypatch.setattr(sidecar_runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sidecar_runtime, "_nvidia_visible", lambda: True)

    assert llamaswap.preferred_llamaswap_engine() == "llamacpp"


def test_llamaswap_engine_prefers_ollama_on_nvidia(monkeypatch):
    from seiso.inference import llamaswap, sidecar_runtime

    monkeypatch.delenv("SEISO_LLAMASWAP_ENGINE", raising=False)
    monkeypatch.setattr(sidecar_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sidecar_runtime, "_nvidia_visible", lambda: True)
    monkeypatch.setattr(sidecar_runtime, "ollama_health_ok", lambda *, url=None: True)

    assert llamaswap.preferred_llamaswap_engine() == "ollama"


def test_llamaswap_engine_falls_back_to_llamacpp_when_ollama_unhealthy(monkeypatch):
    from seiso.inference import llamaswap, sidecar_runtime

    monkeypatch.delenv("SEISO_LLAMASWAP_ENGINE", raising=False)
    monkeypatch.setattr(sidecar_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sidecar_runtime, "_nvidia_visible", lambda: True)
    monkeypatch.setattr(sidecar_runtime, "ollama_health_ok", lambda *, url=None: False)

    assert llamaswap.preferred_llamaswap_engine() == "llamacpp"


def test_llamaswap_engine_override_skips_ollama_health(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.setenv("SEISO_LLAMASWAP_ENGINE", "ollama")
    monkeypatch.setattr(llamaswap, "ollama_health_ok", lambda *, url=None: False)

    assert llamaswap.preferred_llamaswap_engine() == "ollama"


def test_llamaswap_can_be_selected_as_local_backend(tmp_path: Path, monkeypatch):
    from seiso.inference import llamaswap, sidecar_runtime

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    ready = sidecar_runtime.SidecarRuntime(
        available=True, url="http://127.0.0.1:11434", engine="ollama"
    )
    monkeypatch.setattr(llamaswap, "llamaswap_status", lambda: ready)

    assert (
        resolve_local_backend(
            model_path=str(gguf),
            model_format="gguf",
            requested=BACKEND_LLAMASWAP,
        )
        == BACKEND_LLAMASWAP
    )
    assert prepare_model_path(str(gguf), BACKEND_LLAMASWAP) == str(gguf.absolute())


def test_llamaswap_status_requires_reachable_sidecar(monkeypatch):
    from seiso.inference import llamaswap, sidecar_runtime

    monkeypatch.setenv("SEISO_LLAMASWAP_ENABLED", "true")
    monkeypatch.setattr(sidecar_runtime, "ollama_health_ok", lambda *, url=None: False)
    monkeypatch.setattr(sidecar_runtime, "llamaswap_health_ok", lambda *, url=None: False)

    status = llamaswap.llamaswap_status()

    assert status.available is False
    assert status.engine == "llamacpp"
    assert "Neither Ollama" in (status.reason or "")
    assert "SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX" in (status.reason or "")


def test_llamaswap_status_available_when_ollama_healthy(monkeypatch):
    from seiso.inference import llamaswap, sidecar_runtime

    monkeypatch.setenv("SEISO_LLAMASWAP_ENABLED", "true")
    monkeypatch.setattr(sidecar_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sidecar_runtime, "ollama_health_ok", lambda *, url=None: True)
    monkeypatch.setattr(sidecar_runtime, "llamaswap_health_ok", lambda *, url=None: False)
    monkeypatch.setattr(sidecar_runtime, "_nvidia_visible", lambda: True)

    status = llamaswap.llamaswap_status()

    assert status.available is True
    assert status.engine == "ollama"
    assert status.ollama_ready is True


def test_llamaswap_status_does_not_fallback_to_ollama_for_llamacpp_engine(
    monkeypatch,
):
    from seiso.inference import llamaswap, sidecar_runtime

    monkeypatch.setenv("SEISO_LLAMASWAP_ENABLED", "true")
    monkeypatch.setenv("SEISO_LLAMASWAP_ENGINE", "llamacpp")
    monkeypatch.setattr(sidecar_runtime, "ollama_health_ok", lambda *, url=None: True)
    monkeypatch.setattr(sidecar_runtime, "llamaswap_health_ok", lambda *, url=None: False)

    status = llamaswap.llamaswap_status()

    assert status.available is False
    assert status.engine == "llamacpp"
    assert status.ollama_ready is True
    assert status.llamaswap_ready is False


def test_resolve_backend_label_for_sidecar_engine():
    from seiso.inference.backends import BACKEND_LLAMASWAP, resolve_backend_label

    assert resolve_backend_label(BACKEND_LLAMASWAP, sidecar_engine="ollama") == "Ollama sidecar"
    assert resolve_backend_label(BACKEND_LLAMASWAP, sidecar_engine="llamacpp") == "llama-swap"
    assert resolve_backend_label(BACKEND_LLAMASWAP) == "GGUF sidecar"


def test_ollama_registration_available_when_engine_llamacpp(monkeypatch):
    import shutil

    from seiso.inference import sidecar_runtime

    monkeypatch.setenv("SEISO_LLAMASWAP_ENGINE", "llamacpp")
    monkeypatch.setattr(sidecar_runtime, "ollama_health_ok", lambda *, url=None: True)
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/ollama" if name == "ollama" else None,
    )

    assert sidecar_runtime.ollama_registration_available() is True
    assert sidecar_runtime.preferred_sidecar_engine() == "llamacpp"


def test_sidecar_status_prefers_ollama_on_native_linux(monkeypatch):
    from seiso.inference import sidecar_runtime

    monkeypatch.setenv("SEISO_LLAMASWAP_ENABLED", "true")
    monkeypatch.setattr(sidecar_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sidecar_runtime, "_nvidia_visible", lambda: True)
    monkeypatch.setattr(sidecar_runtime, "ollama_health_ok", lambda *, url=None: True)
    monkeypatch.setattr(sidecar_runtime, "llamaswap_health_ok", lambda *, url=None: False)

    status = sidecar_runtime.sidecar_status()
    assert status.available is True
    assert status.engine == "ollama"


def test_create_isolated_gguf_client_prefers_ollama(monkeypatch):
    from seiso.inference import llamaswap, sidecar_runtime
    from seiso.inference.llamaswap import OllamaClient, create_isolated_gguf_client

    monkeypatch.setattr(sidecar_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sidecar_runtime, "ollama_health_ok", lambda *, url=None: True)
    monkeypatch.setattr(sidecar_runtime, "_nvidia_visible", lambda: True)
    monkeypatch.setattr(llamaswap, "ollama_health_ok", lambda *, url=None: True)

    client = create_isolated_gguf_client()
    assert isinstance(client, OllamaClient)


def test_create_isolated_gguf_client_raises_when_ollama_engine_unhealthy(
    monkeypatch,
):
    from seiso.inference import llamaswap
    from seiso.inference.llamaswap import create_isolated_gguf_client

    monkeypatch.setenv("SEISO_LLAMASWAP_ENGINE", "ollama")
    monkeypatch.setattr(llamaswap, "ollama_health_ok", lambda *, url=None: False)

    with pytest.raises(RuntimeError, match="Ollama is not reachable"):
        create_isolated_gguf_client()


def test_ollama_cli_host_matches_seiso_url(monkeypatch):
    from seiso.inference.llamaswap import ollama_cli_host

    monkeypatch.setenv("SEISO_OLLAMA_URL", "http://127.0.0.1:11434")
    assert ollama_cli_host() == "127.0.0.1:11434"
    assert ollama_cli_host(url="http://10.0.0.5:11500") == "10.0.0.5:11500"


def test_llamaswap_request_body_forwards_tools_and_model_override(monkeypatch):
    from seiso.inference.llamaswap import LlamaSwapClient

    monkeypatch.setenv("SEISO_LLAMASWAP_MODEL", "local-qwen")
    client = LlamaSwapClient(url="http://127.0.0.1:8080", engine="llamacpp")

    body = client._request_body(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7,
            "temperature": 0.25,
            "top_p": 0.9,
            "tools_schemas": [{"type": "function", "function": {"name": "search"}}],
        },
        "/tmp/model.gguf",
        stream=True,
    )

    assert body == {
        "model": "local-qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 7,
        "temperature": 0.25,
        "stream": True,
        "top_p": 0.9,
        "tools": [{"type": "function", "function": {"name": "search"}}],
    }


def test_llamaswap_complete_serializes_native_tool_calls(monkeypatch):
    from seiso.inference.llamaswap import LlamaSwapClient

    client = LlamaSwapClient(url="http://127.0.0.1:8080", engine="llamacpp")
    monkeypatch.setattr(
        client,
        "_post_json",
        lambda _path, _body: {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query":"linux"}',
                                }
                            }
                        ]
                    }
                }
            ]
        },
    )

    text = client.complete(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        "/tmp/model.gguf",
    )

    assert '<tool_call>{"name":"search","arguments":{"query":"linux"}}</tool_call>' in text


def test_sidecar_perf_mode_raises_batch_and_keep_alive(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.setenv("SEISO_SIDECAR_PERF_MODE", "1")
    monkeypatch.delenv("SEISO_OLLAMA_NUM_BATCH", raising=False)
    monkeypatch.delenv("SEISO_OLLAMA_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("SEISO_SIDECAR_VRAM_BUDGET_RATIO", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_consumer_nvidia_gpu", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_headroom_mb", lambda: 20_000)

    assert llamaswap.sidecar_ollama_num_batch() == 512
    assert llamaswap.sidecar_ollama_keep_alive() == "10m"
    assert llamaswap._sidecar_vram_budget_ratio() == 0.70


def test_sidecar_num_ctx_buckets_to_prompt_and_generation():
    from seiso.inference.llamaswap import sidecar_num_ctx

    short = [{"role": "user", "content": "hi"}]
    assert sidecar_num_ctx(short, max_tokens=512, ceiling=131072) == 2048

    # ~7500 prompt tokens + 1024 generation lands in the 16384 bucket.
    long = [{"role": "user", "content": "word " * 6000}]
    assert sidecar_num_ctx(long, max_tokens=1024, ceiling=131072) == 16384

    # Never exceed the model's native context ceiling.
    assert sidecar_num_ctx(long, max_tokens=1024, ceiling=8192) == 8192


def test_sidecar_num_ctx_env_override(monkeypatch):
    from seiso.inference.llamaswap import sidecar_num_ctx

    monkeypatch.setenv("SEISO_SIDECAR_NUM_CTX", "32768")
    short = [{"role": "user", "content": "hi"}]
    assert sidecar_num_ctx(short, max_tokens=512, ceiling=131072) == 32768
    assert sidecar_num_ctx(short, max_tokens=512, ceiling=16384) == 16384


def test_plan_sidecar_request_trims_only_at_model_ceiling():
    from seiso.inference.llamaswap import plan_sidecar_request

    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
    }
    messages, num_ctx, max_tokens = plan_sidecar_request(payload, "/tmp/model.gguf")
    assert messages == payload["messages"]
    assert num_ctx == 2048
    assert max_tokens == 256

    # Unknown-model ceiling defaults to 8192; an oversized prompt is trimmed
    # to fit instead of letting the sidecar truncate silently.
    huge = {
        "messages": [{"role": "user", "content": "word " * 60000}],
        "max_tokens": 1024,
    }
    messages, num_ctx, _ = plan_sidecar_request(huge, "/tmp/model.gguf")
    assert num_ctx == 8192
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    assert total_chars < len(huge["messages"][0]["content"])


def test_plan_sidecar_request_treats_n_ctx_as_cap(monkeypatch):
    from seiso.inference.llamaswap import plan_sidecar_request

    monkeypatch.setenv("SEISO_SIDECAR_NUM_CTX", "32768")
    payload = {
        "messages": [{"role": "user", "content": "word " * 6000}],
        "max_tokens": 512,
        "n_ctx": 4096,
    }

    messages, num_ctx, max_tokens = plan_sidecar_request(payload, "/tmp/model.gguf")

    assert num_ctx == 4096
    assert max_tokens == 512
    assert sum(len(str(m.get("content", ""))) for m in messages) < len(
        payload["messages"][0]["content"]
    )


def test_ollama_request_body_uses_native_chat_options(monkeypatch):
    from seiso.inference.llamaswap import OllamaClient

    client = OllamaClient(url="http://127.0.0.1:11434")
    monkeypatch.setenv("SEISO_OLLAMA_NUM_BATCH", "256")
    monkeypatch.setenv("SEISO_OLLAMA_KEEP_ALIVE", "30s")
    monkeypatch.setattr(
        "seiso.inference.llamaswap.sidecar_ollama_num_gpu",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(client, "_resolve_model", lambda model_path, payload: "seiso/test-model")

    body = client._request_body(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 700,
            "temperature": 0.25,
            "top_p": 0.9,
            "tools_schemas": [{"type": "function", "function": {"name": "search"}}],
        },
        "/tmp/model.gguf",
        stream=True,
    )

    assert body == {
        "model": "seiso/test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {
            "num_ctx": 2048,
            "num_predict": 700,
            "temperature": 0.25,
            "num_batch": 256,
            "top_p": 0.9,
        },
        "keep_alive": "30s",
        "tools": [{"type": "function", "function": {"name": "search"}}],
    }


def test_ollama_complete_serializes_native_tool_calls(monkeypatch):
    from seiso.inference.llamaswap import OllamaClient

    client = OllamaClient(url="http://127.0.0.1:11434")
    monkeypatch.setattr(client, "_resolve_model", lambda model_path, payload: "tag")
    monkeypatch.setattr(
        client,
        "_post_json",
        lambda _path, _body: {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": {"query": "linux"},
                        }
                    }
                ],
            }
        },
    )

    text = client.complete(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        "/tmp/model.gguf",
    )

    assert '<tool_call>{"name":"search","arguments":{"query":"linux"}}</tool_call>' in text


def test_sidecar_vram_context_cap_passthrough_off_native_linux(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: False)
    assert llamaswap.sidecar_vram_context_cap("/tmp/model.gguf", 131072) == 131072


def test_sidecar_vram_context_cap_clamps_to_free_vram(monkeypatch):
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_runtime as runtime_mod
    from seiso.inference import llamaswap

    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 6000)
    monkeypatch.setattr(
        runtime_mod,
        "native_linux_llama_context_cap",
        lambda model_path, *, free_mb, n_gpu_layers, ceiling, max_tokens=512: 8192,
    )

    # KV that would fit a 131072 window is bounded to what free VRAM supports.
    assert llamaswap.sidecar_vram_context_cap("/tmp/model.gguf", 131072) == 8192


def test_sidecar_vram_context_cap_forwards_requested_completion(monkeypatch):
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_runtime as runtime_mod
    from seiso.inference import llamaswap

    seen: dict[str, int] = {}
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 6000)

    def fake_cap(model_path, *, free_mb, n_gpu_layers, ceiling, max_tokens=512):
        seen["max_tokens"] = max_tokens
        return 4096 if max_tokens > 512 else int(ceiling)

    monkeypatch.setattr(runtime_mod, "native_linux_llama_context_cap", fake_cap)

    assert llamaswap.sidecar_vram_context_cap("/tmp/model.gguf", 131072, max_tokens=2048) == 4096
    assert seen["max_tokens"] == 2048


def test_sidecar_vram_context_cap_disabled_by_env(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.setenv("SEISO_SIDECAR_VRAM_CLAMP", "0")
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    assert llamaswap.sidecar_vram_context_cap("/tmp/model.gguf", 131072) == 131072


def test_sidecar_ollama_num_gpu_env_override(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.setenv("SEISO_OLLAMA_NUM_GPU", "12")
    # Explicit override wins even off native Linux.
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: False)
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) == 12


def test_sidecar_ollama_num_batch_defaults_on_native_linux(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_BATCH", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_headroom_mb", lambda: 10_000)
    assert llamaswap.sidecar_ollama_num_batch() == 256


def test_sidecar_ollama_num_batch_roomy_includes_consumer_gpus(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_BATCH", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_consumer_nvidia_gpu", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_headroom_mb", lambda: 20_000)
    assert llamaswap.sidecar_ollama_num_batch() == 512


def test_sidecar_ollama_num_batch_reduces_when_headroom_low(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_BATCH", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_headroom_mb", lambda: 6000)
    assert llamaswap.sidecar_ollama_num_batch() == 128


def test_sidecar_ollama_keep_alive_defaults_on_native_linux(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_KEEP_ALIVE", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_headroom_mb", lambda: 10_000)
    assert llamaswap.sidecar_ollama_keep_alive() == "2m"


def test_sidecar_ollama_keep_alive_shortens_when_headroom_low(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_KEEP_ALIVE", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_headroom_mb", lambda: 3000)
    assert llamaswap.sidecar_ollama_keep_alive() == "30s"


def test_sidecar_ollama_num_gpu_none_off_native_linux(monkeypatch):
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: False)
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None


def test_sidecar_ollama_num_gpu_full_offload_when_consumer_residual_ample(monkeypatch):
    """Small models that leave ≥4 GB residual after full offload are not throttled."""
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.delenv("SEISO_OLLAMA_GPU_LAYER_RATIO", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_consumer_nvidia_gpu", lambda: True)
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 24_000)
    monkeypatch.setattr(protection_mod, "discrete_gpu_total_mb", lambda: 24_576)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 3000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)
    monkeypatch.setattr(
        kv_mod,
        "llama_kv_cache_reserve_mb",
        lambda *args, **kwargs: 1000,
    )
    monkeypatch.setattr(
        kv_mod,
        "llama_offload_fits_headroom",
        lambda model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers: True,
    )

    # budget ≈ 0.55 * 24000 = 13200; need = 4000; residual ≈ 9200 ≥ 4096.
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None


def test_sidecar_ollama_num_gpu_full_offload_medium_when_residual_ample(monkeypatch):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.delenv("SEISO_OLLAMA_GPU_LAYER_RATIO", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_consumer_nvidia_gpu", lambda: True)
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 24_000)
    monkeypatch.setattr(protection_mod, "discrete_gpu_total_mb", lambda: 24_576)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 5000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)
    monkeypatch.setattr(
        kv_mod,
        "llama_kv_cache_reserve_mb",
        lambda *args, **kwargs: 1000,
    )
    monkeypatch.setattr(
        kv_mod,
        "llama_offload_fits_headroom",
        lambda model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers: True,
    )

    # budget ≈ 13200; need = 6000; residual ≈ 7200 ≥ 4096 → full offload.
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None


def test_sidecar_ollama_num_gpu_throttles_when_consumer_residual_tight(monkeypatch):
    """When full offload leaves <4 GB residual, footprint layer caps still apply."""
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.delenv("SEISO_OLLAMA_GPU_LAYER_RATIO", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_consumer_nvidia_gpu", lambda: True)
    # free 10000 → budget = min(5500, 4880) = 4880; need = 1700; residual = 3180 < 4096.
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 10_000)
    monkeypatch.setattr(protection_mod, "discrete_gpu_total_mb", lambda: 24_576)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 1200)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)
    monkeypatch.setattr(
        kv_mod,
        "llama_kv_cache_reserve_mb",
        lambda *args, **kwargs: 500,
    )
    monkeypatch.setattr(
        kv_mod,
        "llama_offload_fits_headroom",
        lambda model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers: True,
    )

    # footprint 1700/4880 ≈ 0.35 → small tier → 50% of 32 layers.
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) == 16


def test_sidecar_ollama_gpu_layer_ratio_override_disables_dynamic_cap(monkeypatch):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setenv("SEISO_OLLAMA_GPU_LAYER_RATIO", "1")
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(llamaswap, "_sidecar_consumer_nvidia_gpu", lambda: True)
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 24_000)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 3000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)
    monkeypatch.setattr(
        kv_mod,
        "llama_kv_cache_reserve_mb",
        lambda *args, **kwargs: 1000,
    )
    monkeypatch.setattr(
        kv_mod,
        "llama_offload_fits_headroom",
        lambda model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers: True,
    )

    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None


def test_sidecar_ollama_num_gpu_full_offload_returns_none(monkeypatch):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setenv("SEISO_OLLAMA_GPU_LAYER_RATIO", "1")
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 24000)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 4000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)
    # Full offload (-1) fits comfortably.
    monkeypatch.setattr(
        kv_mod,
        "llama_offload_fits_headroom",
        lambda model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers: True,
    )
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None


def test_sidecar_ollama_num_gpu_partial_offload(monkeypatch):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setenv("SEISO_OLLAMA_GPU_LAYER_RATIO", "1")
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 4000)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 8000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)

    # Only <= 16 layers fit in the reduced free-VRAM budget; full offload OOMs.
    def _fits(model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers):
        if n_gpu_layers == -1:
            return False
        return n_gpu_layers <= 16

    monkeypatch.setattr(kv_mod, "llama_offload_fits_headroom", _fits)
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) == 16


def test_sidecar_ollama_num_gpu_uses_safer_default_vram_budget(monkeypatch):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setenv("SEISO_OLLAMA_GPU_LAYER_RATIO", "1")
    monkeypatch.delenv("SEISO_SIDECAR_VRAM_BUDGET_RATIO", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "discrete_gpu_total_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.hardware.hardware_profile",
        lambda: {"gpus": [{"name": "NVIDIA GeForce RTX 3090"}]},
    )
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 10_000)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 5000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)

    seen: list[int] = []

    def _fits(model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers):
        seen.append(headroom_mb)
        return True

    monkeypatch.setattr(kv_mod, "llama_offload_fits_headroom", _fits)
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None
    assert seen[0] == 4880


def test_sidecar_ollama_num_gpu_keeps_consumer_nvidia_budget_at_32gb(
    monkeypatch,
):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setenv("SEISO_OLLAMA_GPU_LAYER_RATIO", "1")
    monkeypatch.delenv("SEISO_SIDECAR_VRAM_BUDGET_RATIO", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "discrete_gpu_total_mb", lambda: 32768)
    monkeypatch.setattr(
        "seiso.hardware.hardware_profile",
        lambda: {"gpus": [{"name": "NVIDIA GeForce RTX 5090"}]},
    )
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 20_000)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 5000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)

    seen: list[int] = []

    def _fits(model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers):
        seen.append(headroom_mb)
        return True

    monkeypatch.setattr(kv_mod, "llama_offload_fits_headroom", _fits)
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None
    # consumer ratio 0.62 * 20000 free = 12400 (reserve does not bind at 32 GB).
    assert seen[0] == 12_400


def test_sidecar_ollama_num_gpu_budget_ratio_scales_for_larger_nvidia(
    monkeypatch,
):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setenv("SEISO_OLLAMA_GPU_LAYER_RATIO", "1")
    monkeypatch.delenv("SEISO_SIDECAR_VRAM_BUDGET_RATIO", raising=False)
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "discrete_gpu_total_mb", lambda: 49152)
    monkeypatch.setattr(
        "seiso.hardware.hardware_profile",
        lambda: {"gpus": [{"name": "NVIDIA RTX 6000 Ada Generation"}]},
    )
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 20_000)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 5000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)

    seen: list[int] = []

    def _fits(model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers):
        seen.append(headroom_mb)
        return True

    monkeypatch.setattr(kv_mod, "llama_offload_fits_headroom", _fits)
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None
    # 48 GB workstation ratio 0.72 * 20000 free = 14400.
    assert seen[0] == 14_400


def test_sidecar_ollama_num_gpu_budget_ratio_env_override(monkeypatch):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setenv("SEISO_OLLAMA_GPU_LAYER_RATIO", "1")
    monkeypatch.setenv("SEISO_SIDECAR_VRAM_BUDGET_RATIO", "0.9")
    monkeypatch.setenv("SEISO_SIDECAR_VRAM_RESERVE_MB", "0")
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 10_000)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 5000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)

    seen: list[int] = []

    def _fits(model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers):
        seen.append(headroom_mb)
        return True

    monkeypatch.setattr(kv_mod, "llama_offload_fits_headroom", _fits)
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None
    assert seen[0] == 9000


def test_sidecar_ollama_num_gpu_fixed_reserve_env_override(monkeypatch):
    import seiso.inference.backends as backends_mod
    import seiso.memory.protection as protection_mod
    import seiso.memory.protection.llama_kv as kv_mod
    from seiso.inference import llamaswap

    monkeypatch.delenv("SEISO_OLLAMA_NUM_GPU", raising=False)
    monkeypatch.setenv("SEISO_OLLAMA_GPU_LAYER_RATIO", "1")
    monkeypatch.delenv("SEISO_SIDECAR_VRAM_BUDGET_RATIO", raising=False)
    monkeypatch.setenv("SEISO_SIDECAR_VRAM_RESERVE_MB", "7000")
    monkeypatch.setattr(llamaswap, "_sidecar_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(protection_mod, "discrete_gpu_total_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.hardware.hardware_profile",
        lambda: {"gpus": [{"name": "NVIDIA GeForce RTX 4090"}]},
    )
    monkeypatch.setattr(protection_mod, "headroom_mb", lambda: 10_000)
    monkeypatch.setattr(protection_mod, "estimate_path_vram_mb", lambda p: 5000)
    monkeypatch.setattr(backends_mod, "gguf_total_layers", lambda p: 32)

    seen: list[int] = []

    def _fits(model_path, *, headroom_mb, n_gpu_layers, n_ctx, weight_mb, total_layers):
        seen.append(headroom_mb)
        return True

    monkeypatch.setattr(kv_mod, "llama_offload_fits_headroom", _fits)
    assert llamaswap.sidecar_ollama_num_gpu("/tmp/model.gguf", num_ctx=4096) is None
    assert seen[0] == 3000


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)


def test_llamaswap_stream_buffers_fragmented_tool_calls(monkeypatch):
    import urllib.request

    from seiso.inference.llamaswap import LlamaSwapClient

    client = LlamaSwapClient(url="http://127.0.0.1:8080", engine="llamacpp")

    def event(payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n".encode()

    lines = [
        event(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "search", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        event(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"query":'},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        event(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"linux"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        b"data: [DONE]\n",
    ]
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeStreamResponse(lines),
    )

    tokens = list(
        client.stream(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            "/tmp/model.gguf",
            should_stop=lambda: False,
        )
    )

    assert [token.text for token in tokens] == [
        '<tool_call>{"name":"search","arguments":{"query":"linux"}}</tool_call>'
    ]


def test_ollama_stream_parses_native_jsonl(monkeypatch):
    import urllib.request

    from seiso.inference.llamaswap import OllamaClient

    client = OllamaClient(url="http://127.0.0.1:11434")
    monkeypatch.setattr(client, "_resolve_model", lambda model_path, payload: "tag")

    captured: dict[str, str] = {}
    lines = [
        b'{"message": {"content": "Hel"}, "done": false}\n',
        b'{"message": {"content": "lo"}, "done": false}\n',
        b'{"message": {"content": ""}, "done": true}\n',
    ]

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeStreamResponse(lines)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    tokens = [
        token.text
        for token in client.stream(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            "/tmp/model.gguf",
            should_stop=lambda: False,
        )
    ]

    assert tokens == ["Hel", "lo"]
    assert captured["url"].endswith("/api/chat")


def test_ollama_stream_preserves_tool_calls_and_estimates_tokens(monkeypatch):
    import urllib.request

    from seiso.inference.llamaswap import OllamaClient

    client = OllamaClient(url="http://127.0.0.1:11434")
    monkeypatch.setattr(client, "_resolve_model", lambda model_path, payload: "tag")
    lines = [
        json.dumps(
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": {"query": "linux"},
                            }
                        }
                    ]
                },
                "done": False,
            }
        ).encode("utf-8")
        + b"\n",
        b'{"message": {}, "done": true}\n',
    ]
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeStreamResponse(lines),
    )

    tokens = list(
        client.stream(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            "/tmp/model.gguf",
            should_stop=lambda: False,
        )
    )

    assert tokens[0].text.startswith("<tool_call>")
    assert tokens[0].new_tokens > 1


def test_ollama_stream_raises_on_error_event(monkeypatch):
    import urllib.request

    from seiso.inference.llamaswap import OllamaClient

    client = OllamaClient(url="http://127.0.0.1:11434")
    monkeypatch.setattr(client, "_resolve_model", lambda model_path, payload: "tag")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeStreamResponse(
            [b'{"error": "model requires more system memory"}\n']
        ),
    )

    with pytest.raises(RuntimeError, match="more system memory"):
        list(
            client.stream(
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
                "/tmp/model.gguf",
                should_stop=lambda: False,
            )
        )


def test_ollama_client_ensure_ready_checks_health(monkeypatch):
    from seiso.inference import llamaswap
    from seiso.inference.llamaswap import OllamaClient

    monkeypatch.setattr(llamaswap, "ollama_health_ok", lambda *, url=None: False)

    with pytest.raises(RuntimeError, match="Ollama is not reachable"):
        OllamaClient().ensure_ready()


def test_llamaswap_release_external_memory_uses_management_api(monkeypatch):
    from seiso.inference.llamaswap import LlamaSwapClient

    calls: list[tuple[str, str]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append((req.get_method(), req.full_url))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    ok, reason = LlamaSwapClient(url="http://127.0.0.1:8080").release_external_memory(
        "/models/qwen.gguf"
    )

    assert ok is True
    assert reason is None
    assert calls == [("POST", "http://127.0.0.1:8080/api/models/unload")]


@pytest.mark.asyncio
async def test_openai_prepare_payload_passes_through_backend(monkeypatch, tmp_path):
    from forge.api.routes.openai import ChatCompletionRequest, _prepare_openai_chat_payload
    from forge.services import inference_models

    async def fake_list(*_args, **_kwargs):
        return [{"id": "m1", "selectable": True, "format": "gguf", "kind": "local"}]

    async def prepare_llamaswap(*_args, **_kwargs):
        return {
            "model_path": "/tmp/model.gguf",
            "inference_backend": BACKEND_LLAMASWAP,
            "model_format": "gguf",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "hi"}],
        }

    monkeypatch.setattr(inference_models, "list_inference_options", fake_list)
    monkeypatch.setattr("forge.services.openai_chat.prepare_local_chat_target", prepare_llamaswap)

    payload = await _prepare_openai_chat_payload(
        ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
        ),
        "u1",
        object(),
        SimpleNamespace(data_dir=tmp_path),
    )
    assert payload["inference_backend"] == BACKEND_LLAMASWAP


@pytest.mark.asyncio
async def test_openai_prepare_payload_falls_back_to_llamacpp(monkeypatch, tmp_path):
    from forge.api.routes.openai import ChatCompletionRequest, _prepare_openai_chat_payload
    from forge.services import inference_models

    async def fake_list(*_args, **_kwargs):
        return [{"id": "m1", "selectable": True, "format": "gguf", "kind": "local"}]

    async def prepare_llamacpp(*_args, **_kwargs):
        return {
            "model_path": "/tmp/model.gguf",
            "inference_backend": BACKEND_LLAMACPP,
            "model_format": "gguf",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "hi"}],
        }

    monkeypatch.setattr(inference_models, "list_inference_options", fake_list)
    monkeypatch.setattr("forge.services.openai_chat.prepare_local_chat_target", prepare_llamacpp)

    payload = await _prepare_openai_chat_payload(
        ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
        ),
        "u1",
        object(),
        SimpleNamespace(data_dir=tmp_path),
    )
    assert payload["inference_backend"] == BACKEND_LLAMACPP


@pytest.mark.asyncio
async def test_openai_default_model_resolution_reuses_inventory(tmp_path, monkeypatch):
    from forge.api.routes.openai import ChatCompletionRequest, _prepare_openai_chat_payload
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    fallback = tmp_path / "model.safetensors"
    fallback.write_bytes(b"weights")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    await db.add_model(
        user_id="u1",
        name="fallback",
        path=str(fallback),
        format="safetensors",
        size_bytes=fallback.stat().st_size,
    )
    await db.add_model(
        user_id="u1",
        name="model",
        path=str(model),
        format="gguf",
        size_bytes=model.stat().st_size,
    )
    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=True, mlx=False, torch=False),
    )
    monkeypatch.setenv("SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX", "1")
    monkeypatch.setattr(
        "seiso.inference.backends._native_linux_requires_isolated_gguf",
        lambda: False,
    )
    monkeypatch.setattr(
        "forge.services.inference_chat.assert_model_fits_for_load",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "forge.services.inference_chat.assert_backend_runtime_available",
        lambda *_a, **_k: None,
    )

    payload = await _prepare_openai_chat_payload(
        ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
        ),
        "u1",
        db,
        SimpleNamespace(data_dir=tmp_path),
    )

    assert payload["model_path"] == str(model)
    assert payload["model_format"] == "gguf"


@pytest.mark.asyncio
async def test_openai_named_model_resolution_uses_indexed_lookup(tmp_path, monkeypatch):
    from forge.api.routes.openai import ChatCompletionRequest, _prepare_openai_chat_payload
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    row = await db.add_model(
        user_id="u1",
        name="friendly",
        path=str(model),
        format="gguf",
        size_bytes=model.stat().st_size,
    )
    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=True, mlx=False, torch=False),
    )
    monkeypatch.setenv("SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX", "1")
    monkeypatch.setattr(
        "seiso.inference.backends._native_linux_requires_isolated_gguf",
        lambda: False,
    )
    monkeypatch.setattr(
        "forge.services.inference_chat.assert_model_fits_for_load",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "forge.services.inference_chat.assert_backend_runtime_available",
        lambda *_a, **_k: None,
    )

    payload = await _prepare_openai_chat_payload(
        ChatCompletionRequest(
            model=row["id"],
            messages=[{"role": "user", "content": "hi"}],
        ),
        "u1",
        db,
        SimpleNamespace(data_dir=tmp_path),
    )

    assert payload["model_path"] == str(model)
    assert payload["model_format"] == "gguf"


@pytest.mark.asyncio
async def test_runner_routes_tools_to_llamaswap(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen: dict[str, object] = {}

    class FakeClient:
        def complete(self, payload, model_path):
            seen["payload"] = payload
            seen["model_path"] = model_path
            return "tool-ready"

    monkeypatch.setattr(
        runner,
        "_resolve_route",
        lambda _payload, _model_path: ("llamaswap", "/tmp/model.gguf"),
    )
    monkeypatch.setattr(runner._pool, "get_llamaswap", lambda _path, **_kw: FakeClient())
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: 1)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gen: True)
    monkeypatch.setattr(runner, "_ensure_model_switch", AsyncMock())
    monkeypatch.setattr(
        runner,
        "_llamaswap_payload",
        lambda payload, _path: {**payload, "sidecar_active": True},
    )

    result = await runner.chat(
        {
            "model_path": "/tmp/model.gguf",
            "messages": [{"role": "user", "content": "hi"}],
            "tools_schemas": [{"type": "function", "function": {"name": "search"}}],
        }
    )

    assert result == "tool-ready"
    assert seen["model_path"] == "/tmp/model.gguf"
    assert seen["payload"] and seen["payload"]["tools_schemas"]


def test_get_inference_runner_is_singleton():
    import seiso.inference.runner as runner_mod

    first = runner_mod.get_inference_runner()
    second = runner_mod.get_inference_runner()
    assert first is second


def test_inference_orchestrator_uses_shared_runner():
    from pathlib import Path

    import seiso.inference.runner as runner_mod
    from forge.orchestrators.inference import InferenceOrchestrator

    shared = runner_mod.get_inference_runner()
    orchestrator = InferenceOrchestrator(Path("/tmp/seiso-sandbox"))
    assert orchestrator._runner is shared


def test_reset_inference_runtime_creates_fresh_singletons():
    import seiso.inference.runner as runner_mod

    first = runner_mod.get_inference_runner()
    first_pool = first.pool

    runner_mod.reset_inference_runtime(wait=False)
    second = runner_mod.get_inference_runner()

    assert second is not first
    assert second.pool is not first_pool
    runner_mod.reset_inference_runtime(wait=False)


@pytest.mark.asyncio
async def test_local_inference_stream_propagates_errors(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    async def _noop_switch(
        _path: str, *, draft_path: str | None = None, route: str = "llama"
    ) -> None:
        return None

    runner = LocalInferenceRunner()
    monkeypatch.setattr(runner, "_ensure_model_switch", _noop_switch)
    monkeypatch.setattr(
        runner, "_resolve_route", lambda _payload, _path: ("llama", "/tmp/fake.gguf")
    )
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: 1)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gen: True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(runner, "_iter_tokens", _boom)

    with pytest.raises(RuntimeError, match="model load failed"):
        async for _token in runner.stream({"model_path": "/tmp/fake.gguf"}):
            pass


@pytest.mark.asyncio
async def test_local_inference_chat_uses_direct_completion(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    async def _noop_switch(
        _path: str, *, draft_path: str | None = None, route: str = "llama"
    ) -> None:
        return None

    runner = LocalInferenceRunner()
    monkeypatch.setattr(runner, "_ensure_model_switch", _noop_switch)
    monkeypatch.setattr(
        runner, "_resolve_route", lambda _payload, _path: ("llama", "/tmp/fake.gguf")
    )
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: 7)

    calls: list[str] = []

    def _complete(_payload, _path, _route, _generation_id):
        calls.append("complete")
        return "done"

    def _iter_tokens(*_args, **_kwargs):
        raise AssertionError("chat should not use streaming token iteration")

    monkeypatch.setattr(runner, "_complete", _complete)
    monkeypatch.setattr(runner, "_iter_tokens", _iter_tokens)

    assert await runner.chat({"model_path": "/tmp/fake.gguf"}) == "done"
    assert calls == ["complete"]


@pytest.mark.asyncio
async def test_dflash_switch_preserves_warmed_torch_target(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        runner._pool,
        "status",
        lambda: {
            "active_model": "torch:/tmp/target",
            "backend": "torch",
            "path": "/tmp/target",
            "draft_path": None,
        },
    )
    monkeypatch.setattr("seiso.inference.runner.is_dflash_draft", lambda _path: True)
    monkeypatch.setattr(
        runner._pool,
        "prepare_for_load",
        lambda path, backend=None: calls.append((path, backend)),
    )

    await runner._ensure_model_switch("/tmp/target", draft_path="/tmp/dflash.gguf")

    assert calls == [("/tmp/target", BACKEND_TORCH)]


@pytest.mark.asyncio
async def test_torch_speculative_switch_unloads_warmed_single_target(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(
        runner._pool,
        "status",
        lambda: {
            "active_model": "torch:/tmp/target",
            "backend": "torch",
            "path": "/tmp/target",
            "draft_path": None,
        },
    )
    monkeypatch.setattr("seiso.inference.runner.is_dflash_draft", lambda _path: False)
    monkeypatch.setattr(
        runner._pool,
        "prepare_for_load",
        lambda target_path=None, backend=None: calls.append((target_path, backend)),
    )

    await runner._ensure_model_switch("/tmp/target", draft_path="/tmp/draft")

    assert calls == [(None, None)]


@pytest.mark.asyncio
async def test_disable_speculative_unloads_active_bundle(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(
        runner._pool,
        "status",
        lambda: {
            "active_model": "spec:/tmp/target:/tmp/draft",
            "backend": "torch",
            "path": "/tmp/target",
            "draft_path": "/tmp/draft",
        },
    )
    monkeypatch.setattr(
        runner._pool,
        "prepare_for_load",
        lambda target_path=None, backend=None: calls.append((target_path, backend)),
    )

    await runner._ensure_model_switch("/tmp/target")

    assert calls == [(None, None)]


@pytest.mark.asyncio
async def test_dflash_switch_unloads_torch_spec_bundle(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[tuple[str | None, str | None]] = []
    monkeypatch.setattr(
        runner._pool,
        "status",
        lambda: {
            "active_model": "spec:/tmp/target:/tmp/draft",
            "backend": "torch",
            "path": "/tmp/target",
            "draft_path": "/tmp/draft",
        },
    )
    monkeypatch.setattr("seiso.inference.runner.is_dflash_draft", lambda _path: True)
    monkeypatch.setattr(
        runner._pool,
        "prepare_for_load",
        lambda target_path=None, backend=None: calls.append((target_path, backend)),
    )

    await runner._ensure_model_switch("/tmp/target", draft_path="/tmp/dflash.gguf")

    assert calls == [(None, None)]


def test_warm_model_preloads_torch_speculative_pair(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(
        runner,
        "_resolve_route",
        lambda _payload, _path: ("speculative", "/tmp/target"),
    )
    monkeypatch.setattr(
        runner._pool,
        "torch_speculative_pair_fits",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        runner._pool,
        "get_torch_speculative",
        lambda *args, **kwargs: calls.append(("spec", args, kwargs)),
    )
    monkeypatch.setattr(
        runner._pool,
        "get_llama",
        lambda *_args, **_kwargs: pytest.fail("speculative preload used llama path"),
    )

    runner.warm_model({"model_path": "/tmp/target", "draft_model_path": "/tmp/draft"})

    assert calls == [
        (
            "spec",
            ("/tmp/target", "/tmp/draft"),
            {"load_in_4bit": True},
        )
    ]


def test_warm_model_low_memory_preloads_speculative_target_only(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[tuple[str, tuple, dict]] = []
    monkeypatch.setattr(
        runner,
        "_resolve_route",
        lambda _payload, _path: ("speculative", "/tmp/target"),
    )
    monkeypatch.setattr(
        runner._pool,
        "torch_speculative_pair_fits",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        runner._pool,
        "get_torch",
        lambda *args, **kwargs: calls.append(("torch", args, kwargs)),
    )
    monkeypatch.setattr(
        runner._pool,
        "get_torch_speculative",
        lambda *_args, **_kwargs: pytest.fail("speculative pair should not load"),
    )

    runner.warm_model({"model_path": "/tmp/target", "draft_model_path": "/tmp/draft"})

    assert calls == [("torch", ("/tmp/target",), {"load_in_4bit": True})]


def test_warm_model_uses_chat_sized_llama_context(monkeypatch):
    import seiso.inference.runner as runner_mod
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen_ctx: list[int] = []
    trim_ctxs: list[int] = []
    estimates = iter([8192, 4096])

    class FakeLlama:
        _seiso_load_tier = "normal"
        _seiso_n_batch = 128
        _seiso_n_ubatch = 32
        _seiso_n_gpu_layers = -1
        _seiso_load_headroom_mb = 24576
        _seiso_model_path = "/tmp/model.gguf"

    monkeypatch.setattr(runner, "_resolve_route", lambda _payload, path: ("llama", path))
    monkeypatch.setattr(
        runner_mod,
        "estimate_llama_n_ctx",
        lambda *_a, **_k: next(estimates),
    )
    monkeypatch.setattr(
        runner_mod,
        "trim_llama_messages_to_context",
        lambda messages, *, n_ctx, **_k: trim_ctxs.append(int(n_ctx)) or messages,
    )
    monkeypatch.setattr(
        runner._pool,
        "get_llama",
        lambda _path, n_ctx=4096, *, tier="normal", max_tokens=512: (
            seen_ctx.append(n_ctx) or FakeLlama()
        ),
    )
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 128, 32),
    )

    runner.warm_model(
        {
            "model_path": "/tmp/model.gguf",
            "messages": [{"role": "user", "content": "long prompt"}],
            "max_tokens": 128,
        }
    )

    assert seen_ctx == [4096]
    assert trim_ctxs[:2] == [8192, 4096]


def test_warm_model_preloads_dflash_speculative_components(monkeypatch):
    import seiso.inference.runner as runner_mod
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(
        runner,
        "_resolve_route",
        lambda _payload, _path: ("speculative", "/tmp/target"),
    )
    monkeypatch.setattr(runner_mod, "is_dflash_draft", lambda _path: True)
    monkeypatch.setattr(
        runner_mod,
        "get_dflash_draft",
        lambda *args, **kwargs: calls.append(("dflash", args, kwargs)),
    )
    monkeypatch.setattr(
        runner_mod.LocalInferenceRunner,
        "_estimate_dflash_n_ctx",
        staticmethod(lambda _payload, _draft_path: 3072),
    )
    monkeypatch.setattr(
        runner._pool,
        "get_torch",
        lambda *args, **kwargs: calls.append(("torch", args, kwargs)),
    )
    monkeypatch.setattr(
        runner._pool,
        "get_torch_speculative",
        lambda *_args, **_kwargs: pytest.fail("dflash preload should not load torch draft"),
    )

    runner.warm_model({"model_path": "/tmp/target", "draft_model_path": "/tmp/dflash.gguf"})

    assert calls == [
        ("torch", ("/tmp/target",), {"load_in_4bit": True}),
        ("dflash", ("/tmp/dflash.gguf",), {"n_ctx": 3072}),
    ]


def test_dflash_explicit_context_is_clamped(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    monkeypatch.setattr(
        "seiso.inference.context_limits.effective_context_ceiling",
        lambda *_a, **_k: 4096,
    )

    n_ctx = LocalInferenceRunner._estimate_dflash_n_ctx(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
            "n_ctx": 131072,
        },
        "/tmp/dflash.gguf",
    )

    assert n_ctx == 4096


def test_dflash_speculative_stream_loads_draft_with_estimated_context(monkeypatch):
    import seiso.inference.runner as runner_mod
    from seiso.inference.runner import LocalInferenceRunner
    from seiso.inference.streaming import StreamToken

    runner = LocalInferenceRunner()
    calls: list[tuple[str, tuple, dict]] = []

    class _Tokenizer:
        def __call__(self, prompt, **_kwargs):
            return {"input_ids": prompt.split()}

    monkeypatch.setattr(runner_mod, "configure_torch_inference", lambda: None)
    monkeypatch.setattr(runner_mod, "is_dflash_draft", lambda _path: True)
    monkeypatch.setattr(runner_mod, "_native_linux_requires_isolated_gguf", lambda: False)
    monkeypatch.setattr(
        runner._pool,
        "get_torch",
        lambda *_args, **_kwargs: (object(), _Tokenizer()),
    )
    monkeypatch.setattr(
        runner_mod.LocalInferenceRunner,
        "_estimate_dflash_n_ctx",
        staticmethod(lambda _payload, _draft_path: 6144),
    )
    monkeypatch.setattr(
        runner_mod,
        "get_dflash_draft",
        lambda *args, **kwargs: calls.append(("dflash", args, kwargs)) or object(),
    )
    monkeypatch.setattr(
        runner_mod,
        "format_messages_for_prompt",
        lambda _messages, _tokenizer: "prompt",
    )
    monkeypatch.setattr(
        runner_mod,
        "iter_speculative_tokens_dflash",
        lambda **_kwargs: iter([StreamToken("x")]),
    )

    chunks = list(
        runner._torch_speculative_stream(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "draft_model_path": "/tmp/dflash.gguf",
                "max_tokens": 8,
            },
            "/tmp/target",
            should_stop=lambda: False,
        )
    )

    assert [chunk.text for chunk in chunks] == ["x"]
    assert calls == [("dflash", ("/tmp/dflash.gguf",), {"n_ctx": 6144})]


def test_dflash_speculative_blocked_on_native_linux_nvidia(monkeypatch, tmp_path: Path):
    from seiso.inference.runner import LocalInferenceRunner

    target = tmp_path / "target"
    target.mkdir()
    draft = tmp_path / "draft-dflash.gguf"
    _write_arch_gguf(draft, "dflash")

    monkeypatch.setattr(
        "seiso.inference.runner._native_linux_requires_isolated_gguf",
        lambda: True,
    )

    runner = LocalInferenceRunner()
    with pytest.raises(RuntimeError, match="dFlash speculative decoding"):
        runner._resolve_route(
            {"draft_model_path": str(draft)},
            str(target),
        )


@pytest.mark.asyncio
async def test_resolve_dflash_draft_rejected_on_native_linux_nvidia(monkeypatch, tmp_path: Path):
    from fastapi import HTTPException

    from forge.services import inference_chat

    draft = tmp_path / "draft-dflash.gguf"
    _write_arch_gguf(draft, "dflash")

    async def fake_resolve_model_path(*_args, **_kwargs):
        return str(draft)

    from forge.services import inference_chat_draft

    monkeypatch.setattr(
        inference_chat_draft,
        "resolve_model_path",
        fake_resolve_model_path,
    )
    monkeypatch.setattr(
        "seiso.inference.backends._native_linux_requires_isolated_gguf",
        lambda: True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await inference_chat.resolve_draft_model(
            object(),
            "u1",
            SimpleNamespace(data_dir=tmp_path),
            draft_model_id=None,
            draft_model_path=str(draft),
        )

    assert exc_info.value.status_code == 400
    assert "dFlash speculative decoding" in str(exc_info.value.detail)


def test_torch_input_device_prefers_sharded_gpu():
    import torch

    from seiso.inference.runner import LocalInferenceRunner

    class FakeModel:
        hf_device_map = {"embed": "cpu", "layers.0": "cuda:1", "lm_head": "cpu"}

    assert LocalInferenceRunner._torch_input_device(FakeModel()) == torch.device("cuda:1")


def test_torch_input_device_handles_integer_device_map_entries():
    import torch

    from seiso.inference.runner import LocalInferenceRunner

    class FakeModel:
        hf_device_map = {"embed": "cpu", "layers.0": 0, "lm_head": "disk"}

    assert LocalInferenceRunner._torch_input_device(FakeModel()) == torch.device("cuda:0")


def test_torch_input_device_skips_offload_entries_and_falls_back_to_model_device():
    import torch

    from seiso.inference.runner import LocalInferenceRunner

    class FakeModel:
        hf_device_map = {"embed": "cpu", "layers.0": "disk", "lm_head": "meta"}
        device = torch.device("cpu")

    assert LocalInferenceRunner._torch_input_device(FakeModel()) == torch.device("cpu")


def test_torch_stream_propagates_generation_thread_errors(monkeypatch):
    import sys

    import torch

    from seiso.inference.runner import LocalInferenceRunner

    class FakeStreamer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.stopped = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.stopped:
                raise StopIteration
            raise Empty

        def on_finalized_text(self, _text: str, *, stream_end: bool = False) -> None:
            self.stopped = stream_end

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, _prompt: str, return_tensors: str = "pt", **_kwargs):
            ids = [[1, 2, 3]]
            if return_tensors == "pt" and "add_special_tokens" not in _kwargs:
                return {"input_ids": torch.tensor(ids)}
            return {"input_ids": ids}

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **_kwargs):
            raise RuntimeError("generate failed")

    runner = LocalInferenceRunner()
    monkeypatch.setattr(
        runner._pool,
        "get_torch",
        lambda *_args, **_kwargs: (FakeModel(), FakeTokenizer()),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TextIteratorStreamer=FakeStreamer),
    )

    with pytest.raises(RuntimeError, match="generate failed"):
        list(
            runner._torch_stream(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                    "temperature": 0,
                },
                "/tmp/model",
                should_stop=lambda: False,
            )
        )


@pytest.mark.asyncio
async def test_cancel_generation_keeps_loaded_model(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls = {"bump": 0, "unload": 0}

    monkeypatch.setattr(
        runner._pool,
        "status",
        lambda: {"active_model": "m1", "path": "/tmp/model.gguf"},
    )
    monkeypatch.setattr(
        runner._pool,
        "bump_generation",
        lambda: calls.__setitem__("bump", calls["bump"] + 1),
    )
    monkeypatch.setattr(
        runner._pool,
        "unload_all",
        lambda: calls.__setitem__("unload", calls["unload"] + 1),
    )

    status = await runner.cancel_generation()

    assert status["active_model"] == "m1"
    assert calls == {"bump": 1, "unload": 0}


@pytest.mark.asyncio
async def test_list_inference_options_filters_to_installed_backends(monkeypatch, tmp_path):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-q4.gguf"
    model_path.write_bytes(b"gguf")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    await db.add_model(
        user_id="u1",
        name="Model Q4",
        path=str(model_path),
        source="hf:org/Model",
        format="gguf",
        size_bytes=model_path.stat().st_size,
        metadata=_complete_hf_gguf_metadata(model_path.name),
    )

    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=True, mlx=False, torch=False),
    )
    monkeypatch.setattr(
        inference_models,
        "get_gguf_file_size_bytes",
        lambda _repo, _filename: model_path.stat().st_size,
    )

    options = await inference_models.list_inference_options(db, "u1", hardware_aware=False)

    assert options[0]["id"]
    assert options[0]["backends"] == [BACKEND_LLAMACPP]
    assert options[0]["default_backend"] == BACKEND_LLAMACPP


@pytest.mark.asyncio
async def test_list_inference_options_defaults_to_llamaswap_on_nvidia(monkeypatch, tmp_path):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-q4.gguf"
    model_path.write_bytes(b"gguf")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    await db.add_model(
        user_id="u1",
        name="Model Q4",
        path=str(model_path),
        source="hf:org/Model",
        format="gguf",
        size_bytes=model_path.stat().st_size,
        metadata=_complete_hf_gguf_metadata(model_path.name),
    )

    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=True, llamaswap=True, mlx=False, torch=False),
    )
    monkeypatch.setattr(
        inference_models,
        "get_gguf_file_size_bytes",
        lambda _repo, _filename: model_path.stat().st_size,
    )
    monkeypatch.setattr("seiso.inference.llamaswap.llamaswap_enabled", lambda: True)
    monkeypatch.setattr(
        "seiso.inference.backends._native_linux_requires_isolated_gguf",
        lambda: True,
    )

    options = await inference_models.list_inference_options(
        db,
        "u1",
        profile={
            "backend": "cuda",
            "gpus": [{"name": "NVIDIA RTX", "vram_total_mb": 24576}],
            "ram_gb": 32,
        },
    )

    assert options[0]["backends"] == [BACKEND_LLAMASWAP]
    assert options[0]["default_backend"] == BACKEND_LLAMASWAP


@pytest.mark.asyncio
async def test_list_inference_options_does_not_fallback_to_missing_backend(monkeypatch, tmp_path):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-q4.gguf"
    model_path.write_bytes(b"gguf")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    await db.add_model(
        user_id="u1",
        name="Model Q4",
        path=str(model_path),
        source="hf:org/Model",
        format="gguf",
        size_bytes=model_path.stat().st_size,
        metadata=_complete_hf_gguf_metadata(model_path.name),
    )

    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=False, mlx=False, torch=False),
    )
    monkeypatch.setattr(
        inference_models,
        "get_gguf_file_size_bytes",
        lambda _repo, _filename: model_path.stat().st_size,
    )

    options = await inference_models.list_inference_options(db, "u1", hardware_aware=False)

    assert options[0]["backends"] == []
    assert options[0]["default_backend"] == ""


@pytest.mark.asyncio
async def test_list_inference_options_skips_partial_hf_gguf(monkeypatch, tmp_path):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-Q4_K_M.gguf"
    model_path.write_bytes(b"partial")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    await db.add_model(
        user_id="u1",
        name="Model Q4",
        path=str(model_path),
        source="hf:org/Model",
        format="gguf",
        size_bytes=10_000,
        metadata={
            "repo_id": "org/Model",
            "gguf_repo": "mirror/Model-GGUF",
            "gguf_files": ["model-Q4_K_M.gguf"],
        },
    )

    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=True, mlx=False, torch=False),
    )

    options = await inference_models.list_inference_options(db, "u1", hardware_aware=False)

    assert len(options) == 1
    assert options[0]["selectable"] is False
    assert options[0]["status"] == "incomplete"


@pytest.mark.asyncio
async def test_list_inference_options_skips_hf_gguf_without_metadata(monkeypatch, tmp_path):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-Q4_K_M.gguf"
    _write_arch_gguf(model_path, "llama")

    db = Database(tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True)
    await db.add_model(
        user_id="u1",
        name="Model Q4",
        path=str(model_path),
        source="hf:org/Model",
        format="gguf",
        size_bytes=model_path.stat().st_size,
        metadata={},
    )

    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=True, mlx=False, torch=False),
    )

    options = await inference_models.list_inference_options(db, "u1", hardware_aware=False)

    assert len(options) == 1
    assert options[0]["selectable"] is False
    assert options[0]["status"] == "incomplete"


def test_llama_complete_retries_after_inference_oom(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[str] = []

    fail_once = {"remaining": 1}

    class FakeLlama:
        def __init__(self, *, tier: str = "normal") -> None:
            self._seiso_load_tier = tier

        def create_chat_completion(self, **_kwargs):
            if fail_once["remaining"] > 0:
                fail_once["remaining"] -= 1
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return {"choices": [{"message": {"content": "ok"}}]}

    def get_llama(_path, n_ctx=4096, *, tier="normal", max_tokens=512):
        calls.append(f"get:{tier}")
        return FakeLlama(tier=tier)

    def reload_llama(_path, n_ctx, *, tier, batch_override=None, max_tokens=512):
        calls.append(f"reload:{tier}")
        return FakeLlama(tier=tier)

    monkeypatch.setattr(runner._pool, "get_llama", get_llama)
    monkeypatch.setattr(runner._pool, "reload_llama", reload_llama)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gid: True)
    monkeypatch.setattr("seiso.inference.runner.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 512, 128),
    )

    reply = runner._llama_complete(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
        "/tmp/model.gguf",
        generation_id=1,
    )

    assert reply == "ok"
    assert calls == ["get:normal", "reload:compact"]


def test_llama_complete_oom_recovery_passes_batch_override(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen_overrides: list[tuple[int, int] | None] = []
    fail_once = {"remaining": 1}

    class FakeLlama:
        def __init__(self, *, batch: int = 1024, tier: str = "normal") -> None:
            self._seiso_load_tier = tier
            self._seiso_n_batch = batch
            self._seiso_n_ubatch = min(batch, 256)
            self._seiso_last_safe_batch = 512
            self._seiso_last_safe_ubatch = 128

        def create_chat_completion(self, **_kwargs):
            if fail_once["remaining"] > 0:
                fail_once["remaining"] -= 1
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(
        runner._pool,
        "get_llama",
        lambda *_a, **_k: FakeLlama(),
    )

    def reload_llama(_path, n_ctx, *, tier, batch_override=None, max_tokens=512):
        seen_overrides.append(batch_override)
        return FakeLlama(batch=batch_override[0] if batch_override else 512, tier=tier)

    monkeypatch.setattr(runner._pool, "reload_llama", reload_llama)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gid: True)
    monkeypatch.setattr("seiso.inference.runner.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 512, 128),
    )
    monkeypatch.setattr(
        "seiso.memory.protection.discrete_gpu_total_mb",
        lambda _profile=None: 24576,
    )

    reply = runner._llama_complete(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
        "/tmp/model.gguf",
        generation_id=1,
    )

    assert reply == "ok"
    assert seen_overrides == [(64, 32)]


def test_llama_complete_prefill_guard_reloads_before_native_linux_segfault(
    monkeypatch,
):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls: list[str] = []
    seen_overrides: list[tuple[int, int] | None] = []

    class FakeLlama:
        def __init__(self, *, batch: int, tier: str = "normal") -> None:
            self._seiso_load_tier = tier
            self._seiso_n_batch = batch
            self._seiso_n_ubatch = min(batch, 1024)
            self._seiso_n_gpu_layers = -1
            self._seiso_load_headroom_mb = 24576
            self._seiso_model_path = "/tmp/model.gguf"

        def create_chat_completion(self, **_kwargs):
            calls.append(f"complete:{self._seiso_n_batch}")
            return {"choices": [{"message": {"content": "ok"}}]}

    def get_llama(_path, n_ctx=4096, *, tier="normal", max_tokens=512):
        calls.append(f"get:{tier}")
        return FakeLlama(batch=4096, tier=tier)

    def reload_llama(_path, n_ctx, *, tier, batch_override=None, max_tokens=512):
        calls.append(f"reload:{tier}")
        seen_overrides.append(batch_override)
        batch = batch_override[0] if batch_override else 4096
        return FakeLlama(batch=batch, tier=tier)

    monkeypatch.setattr(runner._pool, "get_llama", get_llama)
    monkeypatch.setattr(runner._pool, "reload_llama", reload_llama)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gid: True)
    monkeypatch.setattr("seiso.inference.runner.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (True, 512, 128),
    )

    reply = runner._llama_complete(
        {"messages": [{"role": "user", "content": "x" * 20000}], "max_tokens": 32},
        "/tmp/model.gguf",
        generation_id=1,
    )

    assert reply == "ok"
    assert calls == ["get:normal", "reload:normal", "complete:512"]
    assert seen_overrides == [(512, 128)]


def test_llama_complete_trims_prompt_to_loaded_context(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen_messages: list[list[dict]] = []
    seen_ctx: list[int] = []

    class FakeLlama:
        _seiso_load_tier = "normal"
        _seiso_n_batch = 512
        _seiso_n_ubatch = 128
        _seiso_n_gpu_layers = -1
        _seiso_load_headroom_mb = 24576
        _seiso_model_path = "/tmp/model.gguf"

        def create_chat_completion(self, **kwargs):
            seen_messages.append(kwargs["messages"])
            return {"choices": [{"message": {"content": "ok"}}]}

    def get_llama(_path, n_ctx=4096, *, tier="normal", max_tokens=512):
        seen_ctx.append(n_ctx)
        return FakeLlama()

    monkeypatch.setattr(runner._pool, "get_llama", get_llama)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gid: True)
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 512, 128),
    )

    reply = runner._llama_complete(
        {
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "old " * 10000},
                {"role": "assistant", "content": "history " * 10000},
                {"role": "user", "content": "answer this"},
            ],
            "max_tokens": 128,
            "n_ctx": 2048,
        },
        "/tmp/model.gguf",
        generation_id=1,
    )

    assert reply == "ok"
    assert seen_ctx == [2048]
    assert seen_messages
    assert seen_messages[0][-1]["content"] == "answer this"
    assert len(seen_messages[0]) < 4


def test_llama_complete_recomputes_context_after_prompt_trim(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen_ctx: list[int] = []

    class FakeLlama:
        _seiso_load_tier = "normal"
        _seiso_n_batch = 512
        _seiso_n_ubatch = 128
        _seiso_n_gpu_layers = -1
        _seiso_load_headroom_mb = 24576
        _seiso_model_path = "/tmp/model.gguf"

        def create_chat_completion(self, **_kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

    def get_llama(_path, n_ctx=4096, *, tier="normal", max_tokens=512):
        seen_ctx.append(n_ctx)
        return FakeLlama()

    monkeypatch.setattr(runner._pool, "get_llama", get_llama)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gid: True)
    monkeypatch.setattr(
        "seiso.inference.context_limits.effective_context_ceiling",
        lambda *_a, **_k: 8192,
    )
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 512, 128),
    )

    reply = runner._llama_complete(
        {
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "old " * 10000},
                {"role": "assistant", "content": "history " * 10000},
                {"role": "user", "content": "answer this"},
            ],
            "max_tokens": 128,
        },
        "/tmp/model.gguf",
        generation_id=1,
    )

    assert reply == "ok"
    assert seen_ctx == [2048]


def test_llama_complete_retrims_after_context_recompute(monkeypatch):
    import seiso.inference.runner as runner_mod
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen_ctx: list[int] = []
    trim_ctxs: list[int] = []
    estimates = iter([8192, 4096])

    class FakeLlama:
        _seiso_load_tier = "normal"
        _seiso_n_batch = 512
        _seiso_n_ubatch = 128
        _seiso_n_gpu_layers = -1
        _seiso_load_headroom_mb = 24576
        _seiso_model_path = "/tmp/model.gguf"

        def create_chat_completion(self, **_kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(
        runner_mod,
        "estimate_llama_n_ctx",
        lambda *_a, **_k: next(estimates),
    )
    monkeypatch.setattr(
        runner_mod,
        "trim_llama_messages_to_context",
        lambda messages, *, n_ctx, **_k: trim_ctxs.append(int(n_ctx)) or messages,
    )
    monkeypatch.setattr(
        runner._pool,
        "get_llama",
        lambda _path, n_ctx=4096, *, tier="normal", max_tokens=512: (
            seen_ctx.append(n_ctx) or FakeLlama()
        ),
    )
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gid: True)
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 512, 128),
    )

    reply = runner._llama_complete(
        {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 128},
        "/tmp/model.gguf",
        generation_id=1,
    )

    assert reply == "ok"
    assert seen_ctx == [4096]
    assert trim_ctxs[:2] == [8192, 4096]
    assert all(ctx <= 4096 for ctx in trim_ctxs[1:])


def test_llama_complete_retrims_after_oom_recovery_smaller_context(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    seen_lengths: list[int] = []
    first = {"fail": True}

    class FakeLlama:
        def __init__(self, *, tier: str, actual_ctx: int) -> None:
            self._seiso_load_tier = tier
            self._seiso_n_ctx = actual_ctx
            self._seiso_n_batch = 512
            self._seiso_n_ubatch = 128
            self._seiso_n_gpu_layers = -1
            self._seiso_load_headroom_mb = 24576
            self._seiso_model_path = "/tmp/model.gguf"
            self._seiso_last_safe_batch = 512
            self._seiso_last_safe_ubatch = 128

        def create_chat_completion(self, **kwargs):
            seen_lengths.append(sum(len(str(m.get("content", ""))) for m in kwargs["messages"]))
            if first["fail"]:
                first["fail"] = False
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(
        runner._pool,
        "get_llama",
        lambda *_a, **_k: FakeLlama(tier="normal", actual_ctx=4096),
    )
    monkeypatch.setattr(
        runner._pool,
        "reload_llama",
        lambda *_a, **_k: FakeLlama(tier="compact", actual_ctx=2048),
    )
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gid: True)
    monkeypatch.setattr("seiso.inference.runner.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 512, 128),
    )

    reply = runner._llama_complete(
        {
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "old " * 800},
                {"role": "assistant", "content": "history " * 500},
                {"role": "user", "content": "answer this"},
            ],
            "max_tokens": 128,
            "n_ctx": 4096,
        },
        "/tmp/model.gguf",
        generation_id=1,
    )

    assert reply == "ok"
    assert len(seen_lengths) == 2
    assert seen_lengths[1] < seen_lengths[0]


def test_llama_stream_does_not_retry_after_emitting_text(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    reloads: list[str] = []

    class FakeLlama:
        _seiso_load_tier = "normal"

        def create_chat_completion(self, **_kwargs):
            def _stream():
                yield {"choices": [{"delta": {"content": "hello"}}]}
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

            return _stream()

    monkeypatch.setattr(runner._pool, "get_llama", lambda *_a, **_k: FakeLlama())
    monkeypatch.setattr(
        runner._pool,
        "reload_llama",
        lambda *_a, **kwargs: reloads.append(kwargs.get("tier", "")) or FakeLlama(),
    )

    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 512, 128),
    )

    stream = runner._llama_stream(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
        "/tmp/model.gguf",
        should_stop=lambda: False,
    )

    assert next(stream).text == "hello"
    with pytest.raises(RuntimeError, match="after streaming began"):
        next(stream)
    assert reloads == []


def test_llama_stream_preflight_reloads_before_first_token(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    events: list[tuple[str, object]] = []

    class FakeLlama:
        def __init__(self, *, batch: int, ubatch: int) -> None:
            self._seiso_load_tier = "normal"
            self._seiso_n_ctx = 4096
            self._seiso_n_batch = batch
            self._seiso_n_ubatch = ubatch
            self._seiso_n_gpu_layers = -1
            self._seiso_load_headroom_mb = 24576
            self._seiso_model_path = "/tmp/model.gguf"

        def create_chat_completion(self, **kwargs):
            events.append(("generate", kwargs.get("max_tokens")))
            return iter([{"choices": [{"delta": {"content": "ok"}}]}])

    monkeypatch.setattr(
        runner._pool,
        "get_llama",
        lambda *_a, **_k: FakeLlama(batch=1024, ubatch=512),
    )

    def reload_llama(_path, _n_ctx, *, tier, batch_override=None, max_tokens=512):
        events.append(("reload", batch_override))
        return FakeLlama(batch=batch_override[0], ubatch=batch_override[1])

    monkeypatch.setattr(runner._pool, "reload_llama", reload_llama)
    monkeypatch.setattr("seiso.inference.runner.release_cached_memory", lambda sync=False: None)
    monkeypatch.setattr(
        "seiso.inference.runner.llama_prefill_needs_reload",
        lambda **_kwargs: (False, 1024, 512),
    )
    monkeypatch.setattr(
        "seiso.inference.runner.resolve_llama_decode_budget",
        lambda **_kwargs: SimpleNamespace(
            n_ctx=4096,
            max_tokens=16,
            n_batch=128,
            n_ubatch=64,
            reserve_mb=1024,
            tight=True,
        ),
    )

    chunks = list(
        runner._llama_stream(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128},
            "/tmp/model.gguf",
            should_stop=lambda: False,
        )
    )

    assert [chunk.text for chunk in chunks] == ["ok"]
    assert events == [("reload", (128, 64)), ("generate", 16)]
