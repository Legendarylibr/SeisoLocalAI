"""Per-model prefill (n_batch) and decode (n_ubatch) batch invariants on Linux NVIDIA."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.inference.backends import clear_gguf_caches
from seiso.inference.model_pool import llama_load_kwargs
from seiso.memory.protection import (
    llama_prefill_needs_reload,
    resolve_llama_model_batches,
)


@pytest.fixture(autouse=True)
def _reset_gguf_cache():
    clear_gguf_caches()
    yield
    clear_gguf_caches()


def _write_arch_gguf(
    path: Path, architecture: str, *, extra: list[tuple[bytes, int]] | None = None
) -> None:
    import struct

    arch_key = b"general.architecture"
    arch_value = architecture.encode()
    prefix = architecture.split("-", 1)[0]
    payload = [
        struct.pack("<Q", len(arch_key)),
        arch_key,
        struct.pack("<I", 8),
        struct.pack("<Q", len(arch_value)),
        arch_value,
    ]
    for key, value in extra or []:
        payload.extend(
            [
                struct.pack("<Q", len(key)),
                key,
                struct.pack("<I", 4),
                struct.pack("<I", value),
            ]
        )
    block_key = prefix.encode() + b".block_count"
    payload.extend(
        [
            struct.pack("<Q", len(block_key)),
            block_key,
            struct.pack("<I", 4),
            struct.pack("<I", 32),
        ]
    )
    kv_count = 2 + len(extra or [])
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, kv_count) + b"".join(payload))


MODEL_CASES = [
    pytest.param(
        "gemma3-12b-q4.gguf",
        "gemma3",
        [(b"gemma3.attention.sliding_window", 512)],
        7500,
        False,
        id="gemma3-12b-roomy",
    ),
    pytest.param(
        "gemma3-27b-q4.gguf",
        "gemma3",
        [(b"gemma3.attention.sliding_window", 512)],
        16000,
        True,
        id="gemma3-27b-tight",
    ),
    pytest.param("qwen3-14b-q4.gguf", "qwen3", None, 9000, False, id="qwen3-14b-roomy"),
    pytest.param("llama3-8b-q4.gguf", "llama3", None, 5000, False, id="llama3-8b-roomy"),
    pytest.param(
        "mixtral-8x7b-q4.gguf",
        "mixtral",
        [(b"mixtral.expert_count", 8)],
        26000,
        True,
        id="mixtral-moe-tight",
    ),
]


@pytest.mark.parametrize(
    ("name", "arch", "extra", "weight_mb", "expect_tight"),
    MODEL_CASES,
)
def test_load_kwargs_prefill_decode_invariants_per_model(
    monkeypatch,
    tmp_path: Path,
    name: str,
    arch: str,
    extra: list[tuple[bytes, int]] | None,
    weight_mb: int,
    expect_tight: bool,
):
    import seiso.inference.model_pool as mp

    for env_name in (
        "SEISO_LLAMA_BATCH",
        "SEISO_LLAMA_UBATCH",
        "SEISO_LLAMA_GPU_LAYERS",
    ):
        monkeypatch.delenv(env_name, raising=False)

    gguf = tmp_path / name
    _write_arch_gguf(gguf, arch, extra=extra)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.platform.use_linux_nvidia_inference_guards", lambda **_: True
    )
    monkeypatch.setattr(
        "seiso.memory.protection.seiso_platform.use_linux_nvidia_inference_guards",
        lambda **_: True,
    )
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: weight_mb)
    monkeypatch.setattr(
        "seiso.memory.protection.discrete_gpu_total_mb",
        lambda *_args, **_kwargs: 24576,
    )
    monkeypatch.setattr(
        "seiso.hardware.tiers.discrete_vram_total_mb",
        lambda _profile: 24576,
    )

    kwargs = llama_load_kwargs(4096, model_path=str(gguf))
    batch, ubatch, tight = resolve_llama_model_batches(
        model_path=str(gguf),
        free_mb=24576,
        n_ctx=4096,
        n_gpu_layers=int(
            kwargs["n_gpu_layers"] if kwargs.get("n_gpu_layers") is not None else -1
        ),
        weights_resident=False,
    )

    assert kwargs["n_ubatch"] <= kwargs["n_batch"]
    assert kwargs["offload_kqv"] is False
    assert "op_offload" not in kwargs
    assert ubatch <= batch
    assert tight is expect_tight
    assert kwargs["n_batch"] <= batch
    assert kwargs["n_ubatch"] <= ubatch


@pytest.mark.parametrize(
    ("name", "arch", "extra", "weight_mb", "post_load_free_mb"),
    [
        ("gemma3-12b-q4.gguf", "gemma3", [(b"gemma3.attention.sliding_window", 512)], 7500, 15500),
        ("gemma3-27b-q4.gguf", "gemma3", [(b"gemma3.attention.sliding_window", 512)], 16000, 6500),
        ("qwen3-14b-q4.gguf", "qwen3", None, 9000, 14000),
    ],
)
def test_prefill_guard_keeps_loaded_batches_for_short_prompt(
    monkeypatch,
    tmp_path: Path,
    name: str,
    arch: str,
    extra: list[tuple[bytes, int]] | None,
    weight_mb: int,
    post_load_free_mb: int,
):
    import seiso.inference.model_pool as mp

    for env_name in (
        "SEISO_LLAMA_BATCH",
        "SEISO_LLAMA_UBATCH",
        "SEISO_LLAMA_GPU_LAYERS",
    ):
        monkeypatch.delenv(env_name, raising=False)

    gguf = tmp_path / name
    _write_arch_gguf(gguf, arch, extra=extra)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.platform.use_linux_nvidia_inference_guards", lambda **_: True
    )
    monkeypatch.setattr(
        "seiso.memory.protection.seiso_platform.use_linux_nvidia_inference_guards",
        lambda **_: True,
    )
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {"gpus": [{"vram_total_mb": 24576}]})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: weight_mb)
    monkeypatch.setattr(
        "seiso.memory.protection.discrete_gpu_total_mb",
        lambda *_args, **_kwargs: 24576,
    )
    monkeypatch.setattr(
        "seiso.hardware.tiers.discrete_vram_total_mb",
        lambda _profile: 24576,
    )

    load_kwargs = llama_load_kwargs(4096, model_path=str(gguf))
    loaded_batch = int(load_kwargs["n_batch"])
    loaded_ubatch = int(load_kwargs["n_ubatch"])
    _, _, tight_at_load = resolve_llama_model_batches(
        model_path=str(gguf),
        free_mb=24576,
        n_ctx=4096,
        n_gpu_layers=int(
            load_kwargs["n_gpu_layers"]
            if load_kwargs.get("n_gpu_layers") is not None
            else -1
        ),
        weights_resident=False,
    )

    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: post_load_free_mb)

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hello"}],
        n_ctx=4096,
        loaded_n_batch=loaded_batch,
        loaded_n_ubatch=loaded_ubatch,
        loaded_n_gpu_layers=int(
            load_kwargs["n_gpu_layers"]
            if load_kwargs.get("n_gpu_layers") is not None
            else -1
        ),
        load_tier="normal",
        loaded_headroom_mb=24576,
    )

    assert needs_reload is False
    assert safe_ubatch <= safe_batch
    if tight_at_load:
        assert safe_batch <= loaded_batch
        assert safe_ubatch <= loaded_ubatch
