"""Tests for inference backend selection and model options."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.services.inference_models import resolve_chat_target
from seiso.inference.backends import (
    BACKEND_LLAMACPP,
    BACKEND_OLLAMA,
    BACKEND_TORCH,
    available_backends,
    clear_gguf_caches,
    gguf_architecture,
    match_ollama_name,
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
    assert recommend_backend(model_path=str(gguf), model_format="gguf") == BACKEND_LLAMACPP


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


def test_available_backends_includes_ollama_when_tag_matches(tmp_path: Path):
    gguf = tmp_path / "my-lora.gguf"
    gguf.write_bytes(b"gguf")
    backends = available_backends(
        model_path=str(gguf),
        model_format="gguf",
        ollama_names={"my-lora:latest", "other"},
    )
    assert BACKEND_LLAMACPP in backends
    assert BACKEND_OLLAMA in backends


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


def test_gguf_architecture_reads_metadata(tmp_path: Path):
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf, "llama")

    assert gguf_architecture(str(gguf)) == "llama"


def test_available_backends_rejects_unsupported_dflash_draft(tmp_path: Path):
    gguf = tmp_path / "draft.gguf"
    _write_minimal_gguf(gguf, "dflash-draft")

    assert available_backends(model_path=str(gguf), model_format="gguf", ollama_names=set()) == []


def test_match_ollama_name():
    assert match_ollama_name(
        model_path="/models/foo.gguf",
        model_name="foo",
        ollama_names={"foo:latest"},
    ) == "foo:latest"


def test_resolve_gguf_file_picks_largest(tmp_path: Path):
    small = tmp_path / "small.gguf"
    large = tmp_path / "large.gguf"
    small.write_bytes(b"a")
    large.write_bytes(b"a" * 10)
    assert resolve_gguf_file(str(tmp_path)).name == "large.gguf"


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


def test_resolve_chat_target_ollama_kind():
    option = {
        "id": "ollama:llama3.2",
        "kind": "ollama",
        "name": "llama3.2",
        "ollama_model": "llama3.2",
        "default_backend": BACKEND_OLLAMA,
        "backends": [BACKEND_OLLAMA],
        "path": None,
        "format": None,
    }
    target = resolve_chat_target(option, model_id=None, ollama_model=None, inference_backend="auto")
    assert target["inference_backend"] == BACKEND_OLLAMA
    assert target["ollama_model"] == "llama3.2"


def test_resolve_chat_target_local_gguf_ollama_engine():
    option = {
        "id": "local-1",
        "kind": "local",
        "name": "export-q4",
        "path": "/data/export-q4/model.gguf",
        "format": "gguf",
        "default_backend": BACKEND_LLAMACPP,
        "backends": [BACKEND_LLAMACPP, BACKEND_OLLAMA],
        "ollama_model": "export-q4:latest",
    }
    target = resolve_chat_target(option, model_id="local-1", ollama_model=None, inference_backend="ollama")
    assert target["inference_backend"] == BACKEND_OLLAMA
    assert target["ollama_model"] == "export-q4:latest"


def test_resolve_chat_target_ollama_without_tag_raises():
    option = {
        "id": "local-2",
        "kind": "local",
        "name": "orphan",
        "path": "/data/orphan.gguf",
        "format": "gguf",
        "default_backend": BACKEND_LLAMACPP,
        "backends": [BACKEND_LLAMACPP],
        "ollama_model": None,
    }
    with pytest.raises(ValueError, match="not available in Ollama"):
        resolve_chat_target(option, model_id="local-2", ollama_model=None, inference_backend="ollama")


@pytest.mark.asyncio
async def test_chat_route_keeps_local_gguf_ollama_on_ollama(monkeypatch, tmp_path):
    from forge.api.routes import inference as inference_route

    class FakeDb:
        async def get_thread_for_user(self, *_args, **_kwargs):
            return None

    class FakeOrchestrator:
        def create_job(self, **_kwargs):
            return "job-1"

        async def start(self, job_id, payload):
            assert job_id == "job-1"
            assert payload["inference_backend"] == BACKEND_OLLAMA
            assert payload["ollama_model"] == "export-q4:latest"
            assert not payload.get("model_path")

        async def wait_for(self, _job_id):
            class Job:
                status = type("Status", (), {"value": "completed"})()
                result = {"content": "ok", "backend": BACKEND_OLLAMA}

            return Job()

    async def fake_get_inference_option(*_args, **_kwargs):
        return {
            "id": "local-1",
            "kind": "local",
            "name": "export-q4",
            "path": str(tmp_path / "export-q4.gguf"),
            "format": "gguf",
            "default_backend": BACKEND_LLAMACPP,
            "backends": [BACKEND_LLAMACPP, BACKEND_OLLAMA],
            "ollama_model": "export-q4:latest",
        }

    monkeypatch.setattr(inference_route, "get_inference_option", fake_get_inference_option)

    body = inference_route.ChatRequest(
        model_id="local-1",
        inference_backend=BACKEND_OLLAMA,
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
    )
    result = await inference_route.chat(
        body=body,
        user_id="u1",
        db=FakeDb(),
        orchestrator=FakeOrchestrator(),
        settings=type(
            "Settings",
            (),
            {
                "allow_tools": False,
                "allow_code_exec": False,
                "ollama_base_url": "http://127.0.0.1:11434",
                "data_dir": tmp_path,
            },
        )(),
    )

    assert result["backend"] == BACKEND_OLLAMA


@pytest.mark.asyncio
async def test_local_inference_stream_propagates_errors(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    async def _noop_switch(_path: str, *, draft_path: str | None = None) -> None:
        return None

    runner = LocalInferenceRunner()
    monkeypatch.setattr(runner, "_ensure_model_switch", _noop_switch)
    monkeypatch.setattr(runner, "_resolve_route", lambda _payload, _path: ("llama", "/tmp/fake.gguf"))
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: 1)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gen: True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(runner, "_iter_tokens", _boom)

    with pytest.raises(RuntimeError, match="model load failed"):
        async for _token in runner.stream({"model_path": "/tmp/fake.gguf"}):
            pass


@pytest.mark.asyncio
async def test_cancel_generation_keeps_loaded_model(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls = {"bump": 0, "unload": 0}

    monkeypatch.setattr(runner._pool, "status", lambda: {"active_model": "m1", "path": "/tmp/model.gguf"})
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: calls.__setitem__("bump", calls["bump"] + 1))
    monkeypatch.setattr(runner._pool, "unload_all", lambda: calls.__setitem__("unload", calls["unload"] + 1))

    status = await runner.cancel_generation()

    assert status["active_model"] == "m1"
    assert calls == {"bump": 1, "unload": 0}


@pytest.mark.asyncio
async def test_ollama_chat_unloads_active_local_model(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    calls = {"unload": 0, "ollama": 0}

    async def fake_unload():
        calls["unload"] += 1
        return {"active_model": None}

    async def fake_ollama_chat(*_args, **_kwargs):
        calls["ollama"] += 1
        return "ok"

    monkeypatch.setattr(orchestrator._runner, "cancel_and_unload", fake_unload)
    monkeypatch.setattr(inference_orchestrator, "ollama_chat_completion", fake_ollama_chat)

    result = await orchestrator._ollama_chat(
        {"ollama_model": "llama3.2", "ollama_base_url": "http://127.0.0.1:11434"},
        [{"role": "user", "content": "hello"}],
    )

    assert result == "ok"
    assert calls == {"unload": 1, "ollama": 1}


@pytest.mark.asyncio
async def test_ollama_chat_unloads_previous_ollama_model(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    orchestrator._active_ollama_model = "old-model"
    orchestrator._active_ollama_base_url = "http://127.0.0.1:11434"
    calls: list[tuple[str, str]] = []

    async def fake_unload_model(model: str, base_url: str = ""):
        calls.append(("unload", model))
        assert base_url == "http://127.0.0.1:11434"

    async def fake_cancel_local():
        calls.append(("local", "unload"))
        return {"active_model": None}

    async def fake_ollama_chat(*_args, model: str, **_kwargs):
        calls.append(("chat", model))
        return "ok"

    monkeypatch.setattr(inference_orchestrator, "ollama_unload_model", fake_unload_model)
    monkeypatch.setattr(orchestrator._runner, "cancel_and_unload", fake_cancel_local)
    monkeypatch.setattr(inference_orchestrator, "ollama_chat_completion", fake_ollama_chat)

    result = await orchestrator._ollama_chat(
        {"ollama_model": "new-model", "ollama_base_url": "http://127.0.0.1:11434"},
        [{"role": "user", "content": "hello"}],
    )

    assert result == "ok"
    assert calls == [("local", "unload"), ("unload", "old-model"), ("chat", "new-model")]
    assert orchestrator.active_ollama_model == "new-model"


@pytest.mark.asyncio
async def test_local_chat_unloads_active_ollama_model(monkeypatch, tmp_path):
    from forge.orchestrators import inference as inference_orchestrator

    orchestrator = inference_orchestrator.InferenceOrchestrator(tmp_path)
    orchestrator._active_ollama_model = "llama3.2"
    calls: list[str] = []

    async def fake_unload_model(model: str, base_url: str = ""):
        calls.append(f"ollama:{model}:{base_url}")

    async def fake_local_chat(_payload):
        calls.append("local-chat")
        return "ok"

    monkeypatch.setattr(inference_orchestrator, "ollama_unload_model", fake_unload_model)
    monkeypatch.setattr(orchestrator._runner, "chat", fake_local_chat)

    result = await orchestrator._local_chat({"model_path": "/tmp/model.gguf"})

    assert result == "ok"
    assert calls == ["ollama:llama3.2:", "local-chat"]
    assert orchestrator.active_ollama_model is None


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

    async def empty_ollama_names(*_args, **_kwargs):
        return set()

    monkeypatch.setattr(inference_models, "_ollama_names", empty_ollama_names)
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

    async def empty_ollama_names(*_args, **_kwargs):
        return set()

    monkeypatch.setattr(inference_models, "_ollama_names", empty_ollama_names)
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
        size_bytes=model_path.stat().st_size,
        metadata={
            "repo_id": "org/Model",
            "gguf_repo": "mirror/Model-GGUF",
            "gguf_files": ["model-Q4_K_M.gguf"],
        },
    )

    async def empty_ollama_names(*_args, **_kwargs):
        return set()

    monkeypatch.setattr(inference_models, "_ollama_names", empty_ollama_names)
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

    options = await inference_models.list_inference_options(db, "u1", hardware_aware=False)

    assert options == []


@pytest.mark.asyncio
async def test_list_inference_options_skips_hf_gguf_without_metadata(monkeypatch, tmp_path):
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus

    model_path = tmp_path / "model-Q4_K_M.gguf"
    _write_minimal_gguf(model_path, "llama")

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

    async def empty_ollama_names(*_args, **_kwargs):
        return set()

    monkeypatch.setattr(inference_models, "_ollama_names", empty_ollama_names)
    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=True, mlx=False, torch=False),
    )

    options = await inference_models.list_inference_options(db, "u1", hardware_aware=False)

    assert options == []
