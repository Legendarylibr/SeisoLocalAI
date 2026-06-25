"""Discover pip/system CUDA toolkit for JIT-compiling native kernels."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def _site_packages() -> list[Path]:
    paths: list[Path] = []
    for entry in sys.path:
        p = Path(entry)
        if p.is_dir() and (p / "nvidia").is_dir():
            paths.append(p)
    return paths


@lru_cache(maxsize=1)
def discover_cuda_home() -> str | None:
    """Return CUDA toolkit root containing bin/nvcc, or None."""
    env = os.environ.get("CUDA_HOME", "").strip()
    if env and Path(env, "bin", "nvcc").is_file():
        return env

    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)

    for candidate in ("/usr/local/cuda", "/usr/lib/cuda", "/opt/cuda"):
        if Path(candidate, "bin", "nvcc").is_file():
            return candidate

    for site in _site_packages():
        for rel in ("nvidia/cu13", "nvidia/cu12", "nvidia/cuda"):
            root = site / rel
            if (root / "bin" / "nvcc").is_file():
                return str(root)

    return None


@lru_cache(maxsize=1)
def discover_cccl_include() -> str | None:
    """CCCL headers (nv/target) required by CUDA 13 bf16 headers."""
    try:
        import cuda.cccl.headers as cccl_headers

        include = Path(cccl_headers.__file__).resolve().parent / "include"
        if (include / "nv" / "target").exists():
            return str(include)
    except ImportError:
        pass

    for site in _site_packages():
        candidate = site / "cuda" / "cccl" / "headers" / "include"
        if (candidate / "nv" / "target").exists():
            return str(candidate)
    return None


def configure_cuda_build_env() -> dict[str, str]:
    """
    Set CUDA_HOME/PATH/CC for nvcc JIT builds. Returns metadata for logging.
    """
    meta: dict[str, str] = {}
    home = discover_cuda_home()
    if home:
        os.environ.setdefault("CUDA_HOME", home)
        meta["cuda_home"] = home
        bin_dir = str(Path(home) / "bin")
        path = os.environ.get("PATH", "")
        if bin_dir not in path.split(":"):
            os.environ["PATH"] = f"{bin_dir}:{path}"

    # nvcc defaults to gcc; prefer g++ when gcc is unavailable/restricted.
    if not shutil.which("gcc") and shutil.which("g++"):
        gpp = shutil.which("g++")
        if gpp:
            os.environ.setdefault("CC", gpp)
            os.environ.setdefault("CXX", gpp)
            meta["host_compiler"] = gpp

    cccl = discover_cccl_include()
    if cccl:
        meta["cccl_include"] = cccl

    # Pip wheels ship libcudart.so.NN only; torch cpp_extension always passes -lcudart.
    lib_dir = discover_cudart_lib_dir()
    if lib_dir:
        lib_path = Path(lib_dir)
        link = lib_path / "libcudart.so"
        if not link.exists():
            for name in ("libcudart.so.13", "libcudart.so.12"):
                target = lib_path / name
                if target.is_file():
                    try:
                        link.symlink_to(name)
                        meta["cudart_symlink"] = str(link)
                    except OSError as exc:
                        logger.debug("Could not create libcudart.so symlink: %s", exc)
                    break

    return meta


def cuda_build_include_paths() -> list[str]:
    """Extra -I paths for extension.cpp / nvcc."""
    paths: list[str] = []
    cccl = discover_cccl_include()
    if cccl:
        paths.append(cccl)
    return paths


@lru_cache(maxsize=1)
def discover_cudart_lib_dir() -> str | None:
    """Directory containing libcudart for extension link step."""
    home = discover_cuda_home()
    if home:
        lib = Path(home) / "lib"
        if any((lib / name).is_file() for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12")):
            return str(lib)

    for site in _site_packages():
        for rel in ("nvidia/cuda_runtime/lib", "nvidia/cu13/lib", "nvidia/cu12/lib"):
            lib = site / rel
            if any((lib / name).is_file() for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12")):
                return str(lib)
    return None


def cuda_link_flags() -> list[str]:
    """
    Linker flags for pip CUDA wheels that ship versioned libcudart only
    (e.g. libcudart.so.13 without an unversioned libcudart.so symlink).
    """
    lib_dir = discover_cudart_lib_dir()
    if not lib_dir:
        return []
    lib_path = Path(lib_dir)
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        if (lib_path / name).is_file():
            return [f"-L{lib_dir}", f"-l:{name}"]
    return [f"-L{lib_dir}", "-lcudart"]


def cuda_toolkit_ready() -> bool:
    return discover_cuda_home() is not None


def cuda_toolkit_status() -> dict[str, str | bool | None]:
    home = discover_cuda_home()
    return {
        "cuda_home": home,
        "nvcc": shutil.which("nvcc"),
        "cccl_include": discover_cccl_include(),
        "ready": home is not None,
    }


# Configure nvcc/CUDA_HOME before torch.utils.cpp_extension reads the environment.
configure_cuda_build_env()