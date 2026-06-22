"""Tests for inference backend selection and model options."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.services.inference_models import resolve_chat_target
from seiso.inference.backends import (
    BACKEND_LLAMACPP,
    BACKEND_TORCH,
    available_backends,
    clear_gguf_caches,
    gguf_architecture,
    recommend_backend,
    resolve_gguf_file,
    resolve_local_backend,
)


@pytest.fixture(autouse=True)
def _reset_inference_caches():
    from forge.services import inference_models
    from forge.services.hf_connectivity import InferenceRuntimeStatus, check_inference_runtime

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

    assert available_backends(model_path=str(gguf), model_format="gguf") == []


def test_recommend_backend_detects_extensionless_hf_blob(tmp_path: Path):
    blob = tmp_path / "hf_cache" / "models--org--Model-GGUF" / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    _write_minimal_gguf(blob, "llama")

    assert recommend_backend(model_path=str(blob)) == BACKEND_LLAMACPP
    assert resolve_gguf_file(str(blob)) == blob.absolute()


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
async def test_cancel_generation_keeps_loaded_model(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    runner = LocalInferenceRunner()
    calls = {"bump": 0, "unload": 0}

    monkeypatch.setattr(
        runner._pool, "status", lambda: {"active_model": "m1", "path": "/tmp/model.gguf"}
    )
    monkeypatch.setattr(
        runner._pool, "bump_generation", lambda: calls.__setitem__("bump", calls["bump"] + 1)
    )
    monkeypatch.setattr(
        runner._pool, "unload_all", lambda: calls.__setitem__("unload", calls["unload"] + 1)
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

    monkeypatch.setattr(
        inference_models,
        "check_inference_runtime",
        lambda: InferenceRuntimeStatus(llamacpp=True, mlx=False, torch=False),
    )

    options = await inference_models.list_inference_options(db, "u1", hardware_aware=False)

    assert options == []
