"""Discover pip/system CUDA toolkit for JIT-compiling native kernels."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# CUDA 13 nvcc emits PTX 9.3; cuda-toolkit 13.0.2 wheels shipped ptxas capped at 9.0.
_MIN_PTX_FOR_CUDA_MAJOR = {13: 9.3, 12: 8.0}
_SHADOW_PTXAS_PATH_MARKERS = (
    "triton/backends/nvidia/bin",
    "/torch/bin",
)


def _site_packages() -> list[Path]:
    paths: list[Path] = []
    for entry in sys.path:
        p = Path(entry)
        if p.is_dir() and (p / "nvidia").is_dir():
            paths.append(p)
    return paths


def _nvcc_release_major(home: str) -> int | None:
    nvcc = Path(home) / "bin" / "nvcc"
    if not nvcc.is_file():
        return None
    try:
        out = subprocess.check_output([str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if "release" in line:
            token = line.split("release", 1)[1].strip().split(",", 1)[0].strip()
            return int(token.split(".", 1)[0])
    return None


def _ptxas_max_version(ptxas: str | Path) -> float | None:
    try:
        out = subprocess.check_output(
            [str(ptxas), "--list-version"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    versions: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            versions.append(float(line))
        except ValueError:
            continue
    return max(versions) if versions else None


def _toolkit_ptxas_path(home: str) -> Path | None:
    ptxas = Path(home) / "bin" / "ptxas"
    return ptxas if ptxas.is_file() else None


def _min_ptx_for_toolkit(home: str) -> float:
    major = _nvcc_release_major(home)
    if major is None:
        return 9.0
    return _MIN_PTX_FOR_CUDA_MAJOR.get(major, 8.0)


def toolkit_ptxas_compatible(home: str) -> bool:
    """True when toolkit ptxas can assemble nvcc output for this CUDA major."""
    ptxas = _toolkit_ptxas_path(home)
    if ptxas is None:
        return False
    max_ptx = _ptxas_max_version(ptxas)
    if max_ptx is None:
        return True
    return max_ptx + 1e-9 >= _min_ptx_for_toolkit(home)


def _toolkit_has_device_compiler(home: str) -> bool:
    """Pip/system nvcc needs nvvm/bin/cicc; skip incomplete PATH-only stubs."""
    return (Path(home) / "nvvm" / "bin" / "cicc").is_file()


def _cuda_home_candidates() -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []

    def add(path: str | Path | None) -> None:
        if not path:
            return
        resolved = str(Path(path).resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    env = os.environ.get("CUDA_HOME", "").strip()
    if env and Path(env, "bin", "nvcc").is_file():
        add(env)

    for site in _site_packages():
        for rel in ("nvidia/cu13", "nvidia/cu12", "nvidia/cuda"):
            root = site / rel
            if (root / "bin" / "nvcc").is_file():
                add(root)

    nvcc = shutil.which("nvcc")
    if nvcc:
        add(Path(nvcc).resolve().parent.parent)

    for candidate in ("/usr/local/cuda", "/usr/lib/cuda", "/opt/cuda"):
        if Path(candidate, "bin", "nvcc").is_file():
            add(candidate)

    return candidates


@lru_cache(maxsize=1)
def discover_cuda_home() -> str | None:
    """Return CUDA toolkit root containing bin/nvcc, or None."""
    for home in _cuda_home_candidates():
        if not _toolkit_has_device_compiler(home):
            continue
        if toolkit_ptxas_compatible(home):
            return home
    for home in _cuda_home_candidates():
        if _toolkit_has_device_compiler(home):
            return home
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


def _sanitize_path_for_cuda_build(home: str, path: str) -> str:
    """Keep toolkit bin first; drop stale ptxas shims from Triton/torch."""
    bin_dir = str(Path(home) / "bin")
    kept: list[str] = [bin_dir]
    seen: set[str] = {bin_dir}
    for entry in path.split(":"):
        if not entry or entry in seen:
            continue
        if any(marker in entry for marker in _SHADOW_PTXAS_PATH_MARKERS):
            continue
        seen.add(entry)
        kept.append(entry)
    return ":".join(kept)


def _warn_ptxas_mismatch(home: str) -> None:
    ptxas = _toolkit_ptxas_path(home)
    if ptxas is None:
        return
    max_ptx = _ptxas_max_version(ptxas)
    need = _min_ptx_for_toolkit(home)
    if max_ptx is not None and max_ptx + 1e-9 < need:
        logger.warning(
            "CUDA toolkit ptxas at %s supports PTX %.1f but nvcc needs %.1f+. "
            "Upgrade: pip install 'cuda-toolkit[nvcc]>=13.1.0' "
            "(then rm -rf ~/.cache/torch_extensions/*/seiso_cuda_kernels)",
            ptxas,
            max_ptx,
            need,
        )


def configure_cuda_build_env() -> dict[str, str]:
    """
    Set CUDA_HOME/PATH/CC for nvcc JIT builds. Returns metadata for logging.
    """
    meta: dict[str, str] = {}
    home = discover_cuda_home()
    if home:
        os.environ["CUDA_HOME"] = home
        meta["cuda_home"] = home
        os.environ["PATH"] = _sanitize_path_for_cuda_build(home, os.environ.get("PATH", ""))
        if not toolkit_ptxas_compatible(home):
            _warn_ptxas_mismatch(home)
            meta["ptxas_mismatch"] = "true"

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
    ptxas = str(_toolkit_ptxas_path(home)) if home else None
    max_ptx = _ptxas_max_version(ptxas) if ptxas else None
    need_ptx = _min_ptx_for_toolkit(home) if home else None
    return {
        "cuda_home": home,
        "nvcc": shutil.which("nvcc"),
        "ptxas": ptxas,
        "ptxas_max": max_ptx,
        "ptxas_required": need_ptx,
        "ptxas_compatible": toolkit_ptxas_compatible(home) if home else None,
        "cccl_include": discover_cccl_include(),
        "ready": home is not None and toolkit_ptxas_compatible(home),
    }


# Configure nvcc/CUDA_HOME before torch.utils.cpp_extension reads the environment.
configure_cuda_build_env()