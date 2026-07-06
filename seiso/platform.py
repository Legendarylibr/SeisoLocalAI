"""Host platform detection shared across kernels and security."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import shutil
import site
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CUDA_PRELOAD_LIBS: tuple[str, ...] = (
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
)
# llama-cpp CUDA wheels require cu12; list these before cu13 torch libs on LD_LIBRARY_PATH.
_PRIORITY_NVIDIA_LIB_DIRS: tuple[str, ...] = (
    "cuda_runtime/lib",
    "cublas/lib",
    "cublasLt/lib",
)
_cuda_preloaded = False
_SKIP_AUTO_REPAIR_ENV = "SEISO_SKIP_CUDA_AUTO_REPAIR"


def _add_lib_dir(dirs: list[str], seen: set[str], lib_dir: Path) -> None:
    if not lib_dir.is_dir():
        return
    key = str(lib_dir.resolve())
    if key in seen:
        return
    seen.add(key)
    dirs.append(key)


def pip_nvidia_cuda_lib_dirs() -> list[str]:
    """Return lib directories from pip NVIDIA wheels (cu12 runtime before cu13 torch)."""
    seen: set[str] = set()
    dirs: list[str] = []
    roots: list[Path] = []
    for entry in site.getsitepackages():
        roots.append(Path(entry))
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    roots.append(
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )

    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for rel in _PRIORITY_NVIDIA_LIB_DIRS:
            _add_lib_dir(dirs, seen, nvidia_root / rel)
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            _add_lib_dir(dirs, seen, lib_dir)
        nvvm = nvidia_root / "cuda_nvcc" / "nvvm" / "lib64"
        _add_lib_dir(dirs, seen, nvvm)
    return dirs


def cu12_runtime_installed() -> bool:
    """True when pip ``nvidia-cuda-runtime-cu12`` libs are present."""
    for lib_dir in pip_nvidia_cuda_lib_dirs():
        if (Path(lib_dir) / "libcudart.so.12").is_file():
            return True
    return False


def _auto_repair_enabled() -> bool:
    return os.environ.get(_SKIP_AUTO_REPAIR_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pip_install(*packages: str) -> bool:
    if not packages:
        return True
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *packages],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("pip install failed for %s: %s", packages, exc)
        return False
    if proc.returncode != 0:
        logger.debug(
            "pip install failed for %s: %s",
            packages,
            (proc.stderr or proc.stdout or "").strip()[:500],
        )
    return proc.returncode == 0


def ensure_cu12_runtime_packages(*, auto_install: bool | None = None) -> bool:
    """
    Ensure CUDA 12 runtime wheels exist for llama-cpp-python on Linux NVIDIA hosts.

    PyTorch 2.12 ships cu13 libs; llama-cpp CUDA wheels link ``libcudart.so.12``.
    """
    if platform.system() != "Linux":
        return True
    if cu12_runtime_installed():
        return True
    if auto_install is None:
        auto_install = _auto_repair_enabled()
    if not auto_install:
        return False
    try:
        from seiso.security.nvidia_boundary import nvidia_smi_visible

        if not nvidia_smi_visible():
            return True
    except ImportError:
        return True

    logger.info(
        "Installing nvidia-cuda-runtime-cu12 for llama.cpp GPU offload "
        "(PyTorch cu13 + llama-cpp cu12 split)"
    )
    global _cuda_preloaded
    _cuda_preloaded = False
    if not _pip_install("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"):
        return False
    return cu12_runtime_installed()


def repair_cuda_ptxas_toolkit(*, auto_install: bool | None = None) -> bool:
    """
    Upgrade pip cuda-toolkit when ptxas cannot assemble nvcc PTX 9.3+ output.

    Fresh installs may pull cuda-toolkit 13.0.2 via torch while [cuda] needs >=13.1.0.
    """
    if platform.system() != "Linux":
        return True
    try:
        from seiso.kernels.cuda_env import cuda_toolkit_status

        status = cuda_toolkit_status()
    except ImportError:
        return True
    if status.get("ptxas_compatible"):
        return True
    if auto_install is None:
        auto_install = _auto_repair_enabled()
    if not auto_install:
        return False

    logger.info(
        "Upgrading cuda-toolkit for PTX compatibility (ptxas_max=%s, need=%s)",
        status.get("ptxas_max"),
        status.get("ptxas_required"),
    )
    if not _pip_install("cuda-toolkit[nvcc]>=13.1.0"):
        return False

    try:
        from seiso.kernels.cuda_env import (
            clear_cuda_env_caches,
            cuda_toolkit_status,
        )

        clear_cuda_env_caches()
        status = cuda_toolkit_status()
    except ImportError:
        return True

    if status.get("ptxas_compatible"):
        cache_root = Path.home() / ".cache" / "torch_extensions"
        if cache_root.is_dir():
            for path in cache_root.glob("*/seiso_cuda_kernels"):
                shutil.rmtree(path, ignore_errors=True)
        return True
    return False


def repair_linux_cuda_stack(*, auto_install: bool | None = None) -> dict[str, Any]:
    """Repair common Linux NVIDIA install gaps (cu12 runtime + ptxas toolkit)."""
    if platform.system() != "Linux":
        return {"linux": False, "skipped": True}
    cu12 = ensure_cu12_runtime_packages(auto_install=auto_install)
    ptxas = repair_cuda_ptxas_toolkit(auto_install=auto_install)
    return {
        "linux": True,
        "cu12_runtime": cu12,
        "ptxas_compatible": ptxas,
    }


def preload_cuda_shared_libraries(*, lib_dirs: list[str] | None = None) -> list[str]:
    """Load pip-shipped CUDA runtime libs before ``llama_cpp`` import (Linux dlopen quirk)."""
    global _cuda_preloaded
    if _cuda_preloaded:
        return []
    dirs = lib_dirs if lib_dirs is not None else pip_nvidia_cuda_lib_dirs()
    loaded: list[str] = []
    for lib_name in _CUDA_PRELOAD_LIBS:
        for lib_dir in dirs:
            candidate = Path(lib_dir) / lib_name
            if not candidate.is_file():
                continue
            key = str(candidate.resolve())
            if key in loaded:
                break
            try:
                ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
                loaded.append(key)
                break
            except OSError:
                continue
    # Require libcudart.so.12 when present in preload list — partial load breaks llama.cpp.
    needs_cudart12 = any(
        (Path(d) / "libcudart.so.12").is_file() for d in dirs
    )
    if needs_cudart12:
        _cuda_preloaded = any("libcudart.so.12" in path for path in loaded)
    else:
        _cuda_preloaded = bool(loaded)
    return loaded


def ensure_cuda_library_path() -> list[str]:
    """Prepend pip NVIDIA CUDA libs to ``LD_LIBRARY_PATH`` and preload key shared objects."""
    if platform.system() == "Linux":
        ensure_cu12_runtime_packages()
    dirs = pip_nvidia_cuda_lib_dirs()
    if not dirs:
        return []
    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [p for p in current.split(":") if p]
    merged = dirs + [p for p in current_parts if p not in dirs]
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
    preload_cuda_shared_libraries(lib_dirs=dirs)
    return dirs


def detect_wsl2() -> bool:
    """True when running inside WSL2."""
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl2" in version
