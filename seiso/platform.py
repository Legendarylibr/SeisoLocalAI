"""Host platform detection shared across kernels and security."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def pip_nvidia_cuda_lib_dirs() -> list[str]:
    """Return lib directories from pip ``nvidia-*-cu12`` wheels (user-space CUDA runtime)."""
    seen: set[str] = set()
    dirs: list[str] = []
    roots: list[Path] = []
    for entry in site.getsitepackages():
        roots.append(Path(entry))
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    if hasattr(site, "getusersitepackages"):
        pass
    # Editable installs / venv: also scan sys.prefix site-packages.
    roots.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")

    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in nvidia_root.glob("*/lib"):
            if lib_dir.is_dir():
                key = str(lib_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    dirs.append(key)
        nvvm = nvidia_root / "cuda_nvcc" / "nvvm" / "lib64"
        if nvvm.is_dir():
            key = str(nvvm.resolve())
            if key not in seen:
                seen.add(key)
                dirs.append(key)
    return dirs


def ensure_cuda_library_path() -> list[str]:
    """Prepend pip NVIDIA CUDA libs to ``LD_LIBRARY_PATH`` when present."""
    dirs = pip_nvidia_cuda_lib_dirs()
    if not dirs:
        return []
    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [p for p in current.split(":") if p]
    merged = dirs + [p for p in current_parts if p not in dirs]
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
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
