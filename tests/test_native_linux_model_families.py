"""Native Linux NVIDIA inference safety across popular open-model GGUF families."""

from __future__ import annotations

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
def test_unsafe_partial_offload_for_swa_and_moe_on_linux(
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


def test_native_linux_flash_attn_disabled_for_all_families(monkeypatch, tmp_path: Path):
    import seiso.inference.model_pool as mp

    for arch in ("llama", "qwen2", "gemma3", "deepseek2"):
        monkeypatch.delenv("SEISO_LLAMA_UNSAFE_FLASH_ATTN", raising=False)
        monkeypatch.setenv("SEISO_LLAMA_FLASH_ATTN", "true")
        monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
        monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
        gguf = tmp_path / f"{arch}.gguf"
        _write_arch_gguf(gguf, arch)
        kwargs = llama_load_kwargs(4096, model_path=str(gguf))
        assert "flash_attn" not in kwargs, arch


def test_dense_qwen_allows_partial_while_swa_gemma_falls_back_to_cpu(
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
