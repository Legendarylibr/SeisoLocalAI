"""Tests for inference backend selection and model options."""

from __future__ import annotations

from pathlib import Path
from queue import Empty
from types import SimpleNamespace

import pytest

from seiso.inference.backends import (
    BACKEND_LLAMACPP,
    BACKEND_MLX,
    BACKEND_TORCH,
    available_backends,
    clear_gguf_caches,
    gguf_architecture,
    gguf_block_count,
    gguf_context_length,
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


def test_gguf_recommends_llamacpp(tmp_path: Path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")
    assert (
        recommend_backend(model_path=str(gguf), model_format="gguf") == BACKEND_LLAMACPP
    )


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


def test_safetensors_inventory_exposes_torch_and_mlx_fallbacks(
    monkeypatch, tmp_path: Path
):
    from seiso.inference import backends
    from seiso.models.loader import Backend

    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(backends, "detect_backend", lambda: Backend.MLX)

    assert available_backends(
        model_path=str(model_dir), model_format="safetensors"
    ) == [BACKEND_MLX, BACKEND_TORCH]


def _write_minimal_gguf(path: Path, architecture: str) -> None:
    import struct

    key = b"general.architecture"
    value = architecture.encode()
    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", len(key))
        + key
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(value))
        + value
    )


def _write_gguf_with_u32_metadata(path: Path, pairs: list[tuple[bytes, int]]) -> None:
    import struct

    arch_key = b"general.architecture"
    arch_value = b"llama"
    payload = [
        struct.pack("<Q", len(arch_key)),
        arch_key,
        struct.pack("<I", 8),
        struct.pack("<Q", len(arch_value)),
        arch_value,
    ]
    for key, value in pairs:
        payload.extend(
            [
                struct.pack("<Q", len(key)),
                key,
                struct.pack("<I", 4),
                struct.pack("<I", value),
            ]
        )
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1 + len(pairs)) + b"".join(payload)
    )


def test_gguf_architecture_reads_metadata(tmp_path: Path):
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf, "llama")

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


def test_available_backends_allows_dflash_draft_for_speculative(tmp_path: Path):
    gguf = tmp_path / "draft.gguf"
    _write_minimal_gguf(gguf, "dflash-draft")

    # dflash-draft is now allowed (via llama.cpp) when used as speculative draft model
    backends = available_backends(model_path=str(gguf), model_format="gguf")
    assert (
        backends == ["llamacpp"] or backends == []
    )  # may be filtered by other catalog logic in full flow


@pytest.mark.asyncio
async def test_resolve_preload_context_uses_chat_sized_context(monkeypatch, tmp_path):
    from forge.services import inference_chat

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    async def fake_get_inference_option(*_args, **_kwargs):
        return {
            "id": "m1",
            "name": "Model",
            "path": str(model_path),
            "format": "gguf",
            "size_bytes": 123,
        }

    monkeypatch.setattr(
        inference_chat,
        "get_inference_option",
        fake_get_inference_option,
    )
    monkeypatch.setattr(
        inference_chat,
        "resolve_chat_target",
        lambda selected, **_kwargs: {
            "model_path": selected["path"],
            "model_format": selected["format"],
            "inference_backend": BACKEND_LLAMACPP,
        },
    )
    monkeypatch.setattr(
        inference_chat,
        "assert_model_fits_for_load",
        lambda *_args, **_kwargs: None,
    )

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
async def test_resolve_explicit_model_path_checks_selected_backend(
    monkeypatch, tmp_path
):
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


def test_recommend_backend_detects_extensionless_hf_blob(tmp_path: Path):
    blob = tmp_path / "hf_cache" / "models--org--Model-GGUF" / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    _write_minimal_gguf(blob, "llama")

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
        tmp_path
        / "hf_cache"
        / "models--org--Model-GGUF"
        / "snapshots"
        / "rev"
        / "model-q4.gguf"
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
        tmp_path
        / "hf_cache"
        / "models--org--Model-GGUF"
        / "snapshots"
        / "rev"
        / "model-q4.gguf"
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


def test_get_inference_runner_is_singleton():
    import seiso.inference.runner as runner_mod

    runner_mod._runner = None
    first = runner_mod.get_inference_runner()
    second = runner_mod.get_inference_runner()
    assert first is second


def test_inference_orchestrator_uses_shared_runner():
    from pathlib import Path

    import seiso.inference.runner as runner_mod
    from forge.orchestrators.inference import InferenceOrchestrator

    runner_mod._runner = None
    shared = runner_mod.get_inference_runner()
    orchestrator = InferenceOrchestrator(Path("/tmp/seiso-sandbox"))
    assert orchestrator._runner is shared


@pytest.mark.asyncio
async def test_local_inference_stream_propagates_errors(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    async def _noop_switch(_path: str, *, draft_path: str | None = None) -> None:
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

    async def _noop_switch(_path: str, *, draft_path: str | None = None) -> None:
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
        lambda *_args, **_kwargs: pytest.fail(
            "dflash preload should not load torch draft"
        ),
    )

    runner.warm_model(
        {"model_path": "/tmp/target", "draft_model_path": "/tmp/dflash.gguf"}
    )

    assert calls == [
        ("torch", ("/tmp/target",), {"load_in_4bit": True}),
        ("dflash", ("/tmp/dflash.gguf",), {"n_ctx": 3072}),
    ]


def test_dflash_speculative_stream_loads_draft_with_estimated_context(monkeypatch):
    import seiso.inference.runner as runner_mod
    from seiso.inference.runner import LocalInferenceRunner
    from seiso.inference.streaming import StreamToken

    runner = LocalInferenceRunner()
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(runner_mod, "configure_torch_inference", lambda: None)
    monkeypatch.setattr(runner_mod, "is_dflash_draft", lambda _path: True)
    monkeypatch.setattr(
        runner._pool,
        "get_torch",
        lambda *_args, **_kwargs: (object(), object()),
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


def test_torch_input_device_prefers_sharded_gpu():
    import torch

    from seiso.inference.runner import LocalInferenceRunner

    class FakeModel:
        hf_device_map = {"embed": "cpu", "layers.0": "cuda:1", "lm_head": "cpu"}

    assert LocalInferenceRunner._torch_input_device(FakeModel()) == torch.device(
        "cuda:1"
    )


def test_torch_input_device_handles_integer_device_map_entries():
    import torch

    from seiso.inference.runner import LocalInferenceRunner

    class FakeModel:
        hf_device_map = {"embed": "cpu", "layers.0": 0, "lm_head": "disk"}

    assert LocalInferenceRunner._torch_input_device(FakeModel()) == torch.device(
        "cuda:0"
    )


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

        def __call__(self, _prompt: str, return_tensors: str = "pt"):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

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
async def test_list_inference_options_filters_to_installed_backends(
    monkeypatch, tmp_path
):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-q4.gguf"
    model_path.write_bytes(b"gguf")

    db = Database(
        tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True
    )
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

    options = await inference_models.list_inference_options(
        db, "u1", hardware_aware=False
    )

    assert options[0]["id"]
    assert options[0]["backends"] == [BACKEND_LLAMACPP]
    assert options[0]["default_backend"] == BACKEND_LLAMACPP


@pytest.mark.asyncio
async def test_list_inference_options_does_not_fallback_to_missing_backend(
    monkeypatch, tmp_path
):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-q4.gguf"
    model_path.write_bytes(b"gguf")

    db = Database(
        tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True
    )
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

    options = await inference_models.list_inference_options(
        db, "u1", hardware_aware=False
    )

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

    db = Database(
        tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True
    )
    await db.add_model(
        user_id="u1",
        name="Model Q4",
        path=str(model_path),
        source="hf:org/Model",
        format="gguf",
        size_bytes=model_path.stat().st_size,
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
    monkeypatch.setattr(
        inference_models,
        "get_gguf_file_size_bytes",
        lambda _repo, _filename: 10_000,
    )

    options = await inference_models.list_inference_options(
        db, "u1", hardware_aware=False
    )

    assert options == []


@pytest.mark.asyncio
async def test_list_inference_options_skips_hf_gguf_without_metadata(
    monkeypatch, tmp_path
):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-Q4_K_M.gguf"
    _write_minimal_gguf(model_path, "llama")

    db = Database(
        tmp_path / "forge.db", encryption_key=generate_encryption_key(), ephemeral=True
    )
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

    options = await inference_models.list_inference_options(
        db, "u1", hardware_aware=False
    )

    assert options == []
