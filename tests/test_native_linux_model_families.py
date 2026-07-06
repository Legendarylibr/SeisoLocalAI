"""Native Linux NVIDIA inference safety across popular open-model GGUF families."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from seiso.inference.backends import (
    clear_gguf_caches,
    gguf_is_moe,
    gguf_uses_sliding_window_attention,
)
from seiso.inference.model_pool import (
    _llama_skip_partial_offload,
    fit_llama_gpu_layers,
    llama_load_kwargs,
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


FAMILY_CASES = [
    pytest.param("llama", "llama", False, False, id="llama"),
    pytest.param("qwen2", "qwen2", False, False, id="qwen2"),
    pytest.param("qwen3", "qwen3", False, False, id="qwen3"),
    pytest.param("mistral", "mistral", False, False, id="mistral"),
    pytest.param("gemma3", "gemma3", True, False, id="gemma3-swa"),
    pytest.param("gemma4", "gemma4", True, False, id="gemma4-swa"),
    pytest.param("deepseek2", "deepseek2", False, True, id="deepseek2-moe"),
    pytest.param("qwen2moe", "qwen2moe", False, True, id="qwen2moe"),
    pytest.param("mixtral", "mixtral", False, True, id="mixtral-moe"),
]


@pytest.mark.parametrize(
    ("arch", "prefix", "expect_swa", "expect_moe"),
    FAMILY_CASES,
)
def test_gguf_family_metadata_flags(
    tmp_path: Path, arch: str, prefix: str, expect_swa: bool, expect_moe: bool
):
    extra: list[tuple[bytes, int]] = []
    if expect_swa:
        extra.append((prefix.encode() + b".attention.sliding_window", 512))
    if expect_moe:
        extra.append((prefix.encode() + b".expert_count", 8))
    gguf = tmp_path / f"{arch}.gguf"
    _write_arch_gguf(gguf, arch, extra=extra)

    assert gguf_uses_sliding_window_attention(str(gguf)) is expect_swa
    assert gguf_is_moe(str(gguf)) is expect_moe


UNSAFE_PARTIAL_FAMILIES = [
    ("gemma3", "gemma3", True, False),
    ("gemma4", "gemma4", True, False),
    ("deepseek2", "deepseek2", False, True),
    ("qwen2moe", "qwen2moe", False, True),
    ("mixtral", "mixtral", False, True),
]


@pytest.mark.parametrize(
    ("arch", "prefix", "expect_swa", "expect_moe"),
    UNSAFE_PARTIAL_FAMILIES,
)
def test_swa_and_moe_allow_partial_offload_by_default_on_linux(
    monkeypatch, tmp_path: Path, arch: str, prefix: str, expect_swa: bool, expect_moe: bool
):
    import seiso.inference.model_pool as mp

    extra: list[tuple[bytes, int]] = []
    if expect_swa:
        extra.append((prefix.encode() + b".attention.sliding_window", 512))
    if expect_moe:
        extra.append((prefix.encode() + b".expert_count", 8))
    gguf = tmp_path / f"{arch}.gguf"
    _write_arch_gguf(gguf, arch, extra=extra)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)

    assert _llama_skip_partial_offload(str(gguf)) is False


@pytest.mark.parametrize(
    ("arch", "prefix", "expect_swa", "expect_moe"),
    UNSAFE_PARTIAL_FAMILIES,
)
def test_skip_partial_offload_opt_in_on_linux(
    monkeypatch, tmp_path: Path, arch: str, prefix: str, expect_swa: bool, expect_moe: bool
):
    import seiso.inference.model_pool as mp

    extra: list[tuple[bytes, int]] = []
    if expect_swa:
        extra.append((prefix.encode() + b".attention.sliding_window", 512))
    if expect_moe:
        extra.append((prefix.encode() + b".expert_count", 8))
    gguf = tmp_path / f"{arch}.gguf"
    _write_arch_gguf(gguf, arch, extra=extra)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setenv("SEISO_LLAMA_SKIP_PARTIAL_OFFLOAD", "1")

    assert _llama_skip_partial_offload(str(gguf)) is True


@pytest.mark.parametrize("arch", ["llama", "qwen2", "qwen3", "mistral"])
def test_dense_families_allow_partial_offload_on_linux(
    monkeypatch, tmp_path: Path, arch: str
):
    import seiso.inference.model_pool as mp

    gguf = tmp_path / f"{arch}.gguf"
    _write_arch_gguf(gguf, arch)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)

    assert _llama_skip_partial_offload(str(gguf)) is False


def test_native_linux_flash_attn_family_policy(monkeypatch, tmp_path: Path):
    import seiso.inference.model_pool as mp
    from seiso.inference.backends import clear_gguf_caches

    clear_gguf_caches()
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.platform.use_linux_nvidia_inference_guards", lambda **_: True
    )
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)

    # Default on for all families.
    for arch in ("llama", "qwen2", "qwen3", "mistral", "gemma3", "deepseek2"):
        extra = []
        if arch == "gemma3":
            extra = [(b"gemma3.attention.sliding_window", 512)]
        if arch == "deepseek2":
            extra = [(b"deepseek2.expert_count", 8)]
        gguf = tmp_path / f"{arch}-default.gguf"
        _write_arch_gguf(gguf, arch, extra=extra)
        clear_gguf_caches()
        kwargs = llama_load_kwargs(4096, model_path=str(gguf))
        assert kwargs.get("flash_attn") is True, arch

    # May opt out globally.
    monkeypatch.setenv("SEISO_LLAMA_FLASH_ATTN", "false")
    dense = tmp_path / "llama-off.gguf"
    _write_arch_gguf(dense, "llama")
    assert "flash_attn" not in llama_load_kwargs(4096, model_path=str(dense))


def test_native_linux_kv_quant_dense_opt_in(monkeypatch, tmp_path: Path):
    import sys
    import types

    import seiso.inference.model_pool as mp

    fake_lc = types.SimpleNamespace(GGML_TYPE_Q8_0=8, GGML_TYPE_Q4_K=4)
    fake_mod = types.ModuleType("llama_cpp")
    fake_mod.llama_cpp = fake_lc
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.delenv("SEISO_LLAMA_UNSAFE_KV_QUANT", raising=False)
    monkeypatch.setenv("SEISO_LLAMA_KV_QUANT", "false")
    llama = tmp_path / "llama.gguf"
    _write_arch_gguf(llama, "llama")
    assert mp._llama_kv_quant_options(str(llama)) == [{}]

    monkeypatch.setenv("SEISO_LLAMA_KV_QUANT", "true")
    opts = mp._llama_kv_quant_options(str(llama))
    assert len(opts) == 2  # default + Q8
    assert opts[1]["type_k"] == 8

    gemma = tmp_path / "gemma3.gguf"
    _write_arch_gguf(
        gemma, "gemma3", extra=[(b"gemma3.attention.sliding_window", 512)]
    )
    gemma_opts = mp._llama_kv_quant_options(str(gemma))
    assert len(gemma_opts) == 2
    assert gemma_opts[1]["type_k"] == 8


def test_swa_gemma_allows_partial_offload_like_dense(
    monkeypatch, tmp_path: Path,
):
    import seiso.inference.model_pool as mp

    qwen = tmp_path / "qwen2.gguf"
    gemma = tmp_path / "gemma3.gguf"
    _write_arch_gguf(qwen, "qwen2")
    _write_arch_gguf(
        gemma,
        "gemma3",
        extra=[(b"gemma3.attention.sliding_window", 512)],
    )
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 32)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000
    )
    monkeypatch.setattr(
        "seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: False
    )
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb", lambda *_a, **_k: 512
    )

    def fits(path, **k):
        layers = k.get("n_gpu_layers")
        if layers == -1:
            return False
        return layers == 27

    monkeypatch.setattr(
        "seiso.memory.protection.llama_offload_fits_headroom",
        lambda path, **k: fits(str(path), **k),
    )

    assert fit_llama_gpu_layers(str(qwen), -1, 12000, n_ctx=4096) == 27
    assert fit_llama_gpu_layers(str(gemma), -1, 12000, n_ctx=4096) == 27
