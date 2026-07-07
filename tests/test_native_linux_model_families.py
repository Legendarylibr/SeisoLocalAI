"""Native Linux NVIDIA inference safety across popular open-model GGUF families."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from gguf_fixtures import write_arch_gguf as _write_arch_gguf

from seiso.inference.backends import (
    gguf_architecture,
    gguf_is_moe,
    gguf_uses_sliding_window_attention,
)
from seiso.inference.family_policy import policy_for_gguf
from seiso.inference.model_pool import (
    _llama_skip_partial_offload,
    fit_llama_gpu_layers,
    llama_load_kwargs,
)

FAMILY_CASES = [
    pytest.param("llama", "llama", False, False, id="llama"),
    pytest.param("llama3", "llama3", False, False, id="llama3"),
    pytest.param("qwen2", "qwen2", False, False, id="qwen2"),
    pytest.param("qwen3", "qwen3", False, False, id="qwen3"),
    pytest.param("mistral", "mistral", False, False, id="mistral"),
    pytest.param("gemma2", "gemma2", False, False, id="gemma2"),
    pytest.param("phi3", "phi3", False, False, id="phi3"),
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


def test_gguf_metadata_preserves_architecture_before_unknown_value(tmp_path: Path):
    import struct

    gguf = tmp_path / "gemma3-new-metadata.gguf"
    arch_key = b"general.architecture"
    arch_value = b"gemma3"
    unknown_key = b"gemma3.future_metadata"
    payload = [
        struct.pack("<Q", len(arch_key)),
        arch_key,
        struct.pack("<I", 8),
        struct.pack("<Q", len(arch_value)),
        arch_value,
        struct.pack("<Q", len(unknown_key)),
        unknown_key,
        struct.pack("<I", 99),
    ]
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + b"".join(payload))

    assert gguf_architecture(str(gguf)) == "gemma3"


@pytest.mark.parametrize(
    ("name", "expect_kind", "expect_partial"),
    [
        ("Gemma-3-27B-Q4_K_M.gguf", "swa", False),
        ("gemma3n-e4b-it-Q4_K_M.gguf", "swa", False),
        ("Mixtral-8x7B-Q4_K_M.gguf", "moe", True),
        ("Qwen3MoE-30B-A3B-Q4_K_M.gguf", "moe", True),
    ],
)
def test_family_policy_uses_filename_hints_when_metadata_unreadable(
    tmp_path: Path, name: str, expect_kind: str, expect_partial: bool
):
    gguf = tmp_path / name
    gguf.write_bytes(b"GGUF")

    policy = policy_for_gguf(str(gguf))

    assert policy.kind == expect_kind
    assert policy.allow_partial_offload is expect_partial


def test_swa_speed_extras_disable_swa_full_by_default(monkeypatch, tmp_path: Path):
    import seiso.inference.model_pool as mp

    gemma = tmp_path / "gemma3.gguf"
    llama = tmp_path / "llama.gguf"
    _write_arch_gguf(
        gemma,
        "gemma3",
        extra=[(b"gemma3.attention.sliding_window", 512)],
    )
    _write_arch_gguf(llama, "llama")

    assert mp._llama_speed_extras(str(gemma)) == {"swa_full": False}
    assert mp._llama_speed_extras(str(llama)) == {}

    monkeypatch.setenv("SEISO_LLAMA_SWA_FULL", "true")
    assert mp._llama_speed_extras(str(gemma)) == {}


@pytest.mark.parametrize(
    ("arch", "prefix", "expect_kind", "expect_partial"),
    [
        ("llama3", "llama3", "dense", True),
        ("qwen3", "qwen3", "dense", True),
        ("gemma2", "gemma2", "dense", True),
        ("phi3", "phi3", "dense", True),
        ("mistral", "mistral", "dense", True),
        ("gemma3", "gemma3", "swa", False),
        ("gemma4", "gemma4", "swa", False),
        ("mixtral", "mixtral", "moe", True),
        ("qwen2moe", "qwen2moe", "moe", True),
        ("deepseek2", "deepseek2", "moe", True),
    ],
)
def test_gguf_family_policy_matrix(
    tmp_path: Path, arch: str, prefix: str, expect_kind: str, expect_partial: bool
):
    extra: list[tuple[bytes, int]] = []
    if expect_kind == "swa":
        extra.append((prefix.encode() + b".attention.sliding_window", 512))
    if expect_kind == "moe":
        extra.append((prefix.encode() + b".expert_count", 8))
    gguf = tmp_path / f"{arch}.gguf"
    _write_arch_gguf(gguf, arch, extra=extra)

    policy = policy_for_gguf(str(gguf))

    assert policy.kind == expect_kind
    assert policy.allow_partial_offload is expect_partial


SWA_PARTIAL_FAMILIES = [
    ("gemma3", "gemma3", True, False),
    ("gemma4", "gemma4", True, False),
]

MOE_PARTIAL_FAMILIES = [
    ("deepseek2", "deepseek2", False, True),
    ("qwen2moe", "qwen2moe", False, True),
    ("mixtral", "mixtral", False, True),
]


@pytest.mark.parametrize(
    ("arch", "prefix", "expect_swa", "expect_moe"),
    SWA_PARTIAL_FAMILIES,
)
def test_swa_blocks_partial_offload_on_linux(
    monkeypatch, tmp_path: Path, arch: str, prefix: str, expect_swa: bool, expect_moe: bool
):
    import seiso.inference.model_pool as mp

    extra: list[tuple[bytes, int]] = []
    if expect_swa:
        extra.append((prefix.encode() + b".attention.sliding_window", 512))
    gguf = tmp_path / f"{arch}.gguf"
    _write_arch_gguf(gguf, arch, extra=extra)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)

    assert _llama_skip_partial_offload(str(gguf)) is True


@pytest.mark.parametrize(
    ("arch", "prefix", "expect_swa", "expect_moe"),
    MOE_PARTIAL_FAMILIES,
)
def test_moe_allows_partial_offload_on_linux(
    monkeypatch, tmp_path: Path, arch: str, prefix: str, expect_swa: bool, expect_moe: bool
):
    import seiso.inference.model_pool as mp

    extra: list[tuple[bytes, int]] = []
    if expect_moe:
        extra.append((prefix.encode() + b".expert_count", 8))
    gguf = tmp_path / f"{arch}.gguf"
    _write_arch_gguf(gguf, arch, extra=extra)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)

    assert _llama_skip_partial_offload(str(gguf)) is False


@pytest.mark.parametrize(
    ("arch", "prefix", "expect_swa", "expect_moe"),
    SWA_PARTIAL_FAMILIES,
)
def test_unsafe_partial_offload_allows_swa_on_linux(
    monkeypatch, tmp_path: Path, arch: str, prefix: str, expect_swa: bool, expect_moe: bool
):
    import seiso.inference.model_pool as mp

    extra: list[tuple[bytes, int]] = []
    if expect_swa:
        extra.append((prefix.encode() + b".attention.sliding_window", 512))
    gguf = tmp_path / f"{arch}.gguf"
    _write_arch_gguf(gguf, arch, extra=extra)
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setenv("SEISO_LLAMA_UNSAFE_PARTIAL_OFFLOAD", "1")

    assert _llama_skip_partial_offload(str(gguf)) is False


@pytest.mark.parametrize("arch", ["llama", "llama3", "qwen2", "qwen3", "mistral", "gemma2", "phi3"])
def test_dense_families_allow_partial_offload_on_linux(monkeypatch, tmp_path: Path, arch: str):
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
    monkeypatch.setattr("seiso.platform.use_linux_nvidia_inference_guards", lambda **_: True)
    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SEISO_LLAMA_UNSAFE_FLASH_ATTN", raising=False)

    # Default off even for dense families.
    monkeypatch.setenv("SEISO_LLAMA_FLASH_ATTN", "false")
    dense = tmp_path / "llama.gguf"
    _write_arch_gguf(dense, "llama")
    assert "flash_attn" not in llama_load_kwargs(4096, model_path=str(dense))

    # Dense may opt in via SEISO_LLAMA_FLASH_ATTN=true.
    monkeypatch.setenv("SEISO_LLAMA_FLASH_ATTN", "true")
    for arch in ("llama", "llama3", "qwen2", "qwen3", "mistral", "gemma2", "phi3"):
        gguf = tmp_path / f"{arch}-optin.gguf"
        _write_arch_gguf(gguf, arch)
        clear_gguf_caches()
        kwargs = llama_load_kwargs(4096, model_path=str(gguf))
        assert kwargs.get("flash_attn") is True, arch

    # MoE / SWA stay blocked without UNSAFE.
    for arch, extra in (
        ("gemma3", [(b"gemma3.attention.sliding_window", 512)]),
        ("deepseek2", [(b"deepseek2.expert_count", 8)]),
    ):
        gguf = tmp_path / f"{arch}-blocked.gguf"
        _write_arch_gguf(gguf, arch, extra=extra)
        clear_gguf_caches()
        kwargs = llama_load_kwargs(4096, model_path=str(gguf))
        assert "flash_attn" not in kwargs, arch


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
    _write_arch_gguf(gemma, "gemma3", extra=[(b"gemma3.attention.sliding_window", 512)])
    assert mp._llama_kv_quant_options(str(gemma)) == [{}]


def test_dense_qwen_allows_partial_while_swa_gemma_falls_back_to_cpu(
    monkeypatch,
    tmp_path: Path,
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
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)
    monkeypatch.setattr("seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: False)
    monkeypatch.setattr("seiso.memory.protection.llama_kv_cache_reserve_mb", lambda *_a, **_k: 512)

    def fits(path, **k):
        layers = k.get("n_gpu_layers")
        if "gemma" in str(path):
            return layers == 0
        if layers == -1:
            return False
        return layers == 27

    monkeypatch.setattr(
        "seiso.memory.protection.llama_offload_fits_headroom",
        lambda path, **k: fits(str(path), **k),
    )

    assert fit_llama_gpu_layers(str(qwen), -1, 12000, n_ctx=4096) == 27
    assert fit_llama_gpu_layers(str(gemma), -1, 12000, n_ctx=4096) == 0


def test_swa_gemma_uses_full_gpu_when_it_fits(monkeypatch, tmp_path: Path):
    import seiso.inference.model_pool as mp

    gemma = tmp_path / "gemma3.gguf"
    _write_arch_gguf(
        gemma,
        "gemma3",
        extra=[(b"gemma3.attention.sliding_window", 512)],
    )
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 32)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)
    monkeypatch.setattr("seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: False)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_offload_fits_headroom",
        lambda _path, **k: k.get("n_gpu_layers") == -1,
    )

    assert fit_llama_gpu_layers(str(gemma), -1, 24576, n_ctx=4096) == -1


def test_qwen36_27b_24gb_uses_full_offload_when_estimate_fits(monkeypatch, tmp_path: Path):
    import seiso.inference.model_pool as mp

    qwen = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    _write_arch_gguf(qwen, "qwen3")
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 64)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 17_000)
    monkeypatch.setattr("seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: False)
    monkeypatch.setattr("seiso.memory.protection.llama_kv_cache_reserve_mb", lambda *_a, **_k: 1024)

    def fits(_path, **kwargs):
        layers = kwargs.get("n_gpu_layers")
        budget = int(kwargs.get("headroom_mb") or 0)
        if layers == -1:
            return budget >= 18_024
        return layers <= 48

    monkeypatch.setattr("seiso.memory.protection.llama_offload_fits_headroom", fits)

    layers = fit_llama_gpu_layers(str(qwen), -1, 24_576, n_ctx=4096)

    assert layers == -1


def test_qwen36_27b_24gb_uses_partial_offload_when_full_does_not_fit(monkeypatch, tmp_path: Path):
    import seiso.inference.model_pool as mp

    qwen = tmp_path / "Qwen3.6-27B-UD-Q4_K_XL.gguf"
    _write_arch_gguf(qwen, "qwen3")
    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 64)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 17_000)
    monkeypatch.setattr("seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: True)
    monkeypatch.setattr("seiso.memory.protection.llama_kv_cache_reserve_mb", lambda *_a, **_k: 4096)

    def fits(_path, **kwargs):
        layers = kwargs.get("n_gpu_layers")
        if layers == -1:
            return False
        return layers <= 48

    monkeypatch.setattr("seiso.memory.protection.llama_offload_fits_headroom", fits)

    layers = fit_llama_gpu_layers(str(qwen), -1, 24_576, n_ctx=8192)

    assert 0 < layers <= 48
