"""CUDA preload helpers for native Linux llama.cpp import."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import seiso.platform as platform_mod


@pytest.fixture(autouse=True)
def _reset_cuda_preload_state():
    platform_mod.reset_cuda_preload_state()
    yield
    platform_mod.reset_cuda_preload_state()


def test_preload_retries_when_first_pass_loaded_nothing(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    loaded = platform_mod.preload_cuda_shared_libraries(lib_dirs=[str(empty_dir)])

    assert loaded == []
    assert platform_mod._cuda_preloaded is False

    lib_dir = tmp_path / "nvidia" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libcudart.so.12").write_bytes(b"\x00")
    monkeypatch.setattr(platform_mod.ctypes, "CDLL", MagicMock())

    loaded = platform_mod.preload_cuda_shared_libraries(lib_dirs=[str(lib_dir)])

    assert len(loaded) == 1
    assert platform_mod._cuda_preloaded is True


def test_preload_falls_back_to_family_when_dt_needed_soname_missing(
    tmp_path, monkeypatch
):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "libcudart.so.12").write_bytes(b"\x00")
    monkeypatch.setattr(platform_mod.ctypes, "CDLL", MagicMock())
    monkeypatch.setattr(
        platform_mod,
        "required_cuda_sonames",
        lambda: ["libcudart.so.99"],
    )

    loaded = platform_mod.preload_cuda_shared_libraries(lib_dirs=[str(lib_dir)])

    assert any("libcudart.so.12" in path for path in loaded)


def test_required_cuda_sonames_inserts_cublaslt_before_cublas(
    tmp_path, monkeypatch
):
    lib_dir = tmp_path / "llama_cpp" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libllama.so").write_bytes(b"\x7fELF")

    monkeypatch.setattr(
        platform_mod,
        "_llamacpp_lib_dirs",
        lambda: [lib_dir],
    )
    monkeypatch.setattr(
        platform_mod,
        "_elf_needed_sonames",
        lambda _path: ["libcudart.so.12", "libcublas.so.12"],
    )

    names = platform_mod.required_cuda_sonames()

    assert names.index("libcublasLt.so.12") < names.index("libcublas.so.12")


def test_reset_cuda_preload_state_allows_retry(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    platform_mod.preload_cuda_shared_libraries(lib_dirs=[str(empty_dir)])
    assert platform_mod._cuda_preloaded is False

    platform_mod.reset_cuda_preload_state()
    assert platform_mod._cuda_preloaded is False

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "libcudart.so.12").write_bytes(b"\x00")
    monkeypatch.setattr(platform_mod.ctypes, "CDLL", MagicMock())
    loaded = platform_mod.preload_cuda_shared_libraries(lib_dirs=[str(lib_dir)])
    assert len(loaded) == 1
