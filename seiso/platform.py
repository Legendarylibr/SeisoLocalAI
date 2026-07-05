"""Host platform detection shared across kernels and security."""

from __future__ import annotations

import ctypes
import os
import site
import struct
import sys
from pathlib import Path

# CUDA library families the llama-cpp-python CUDA wheels link against.
_CUDA_PRELOAD_FAMILIES: tuple[str, ...] = ("libcudart", "libcublasLt", "libcublas")
# Known CUDA runtime majors shipped by pip nvidia wheels (cu12 first — it
# matches every published llama-cpp-python CUDA wheel today).
_CUDA_PRELOAD_VERSIONS: tuple[str, ...] = ("12", "13")
_CUDA_SONAME_PREFIXES: tuple[str, ...] = (
    "libcudart.so.",
    "libcublas.so.",
    "libcublasLt.so.",
)
_cuda_preloaded = False


def _elf_needed_sonames(path: Path) -> list[str]:
    """DT_NEEDED sonames from a 64-bit little-endian ELF (header seeks only)."""
    try:
        with path.open("rb") as f:
            ident = f.read(16)
            if len(ident) < 16 or ident[:4] != b"\x7fELF":
                return []
            if ident[4] != 2 or ident[5] != 1:  # ELFCLASS64, little-endian only
                return []
            f.seek(0x28)
            (e_shoff,) = struct.unpack("<Q", f.read(8))
            f.seek(0x3A)
            e_shentsize, e_shnum = struct.unpack("<HH", f.read(4))
            if not e_shoff or not e_shnum or e_shentsize < 64:
                return []
            f.seek(e_shoff)
            raw = f.read(e_shentsize * e_shnum)
            headers = [
                struct.unpack_from("<IIQQQQIIQQ", raw, i * e_shentsize) for i in range(e_shnum)
            ]
            dynamic = next((sh for sh in headers if sh[1] == 6), None)  # SHT_DYNAMIC
            if dynamic is None or dynamic[6] >= len(headers):
                return []
            strtab = headers[dynamic[6]]  # sh_link -> .dynstr
            f.seek(strtab[4])
            strdata = f.read(strtab[5])
            f.seek(dynamic[4])
            dyndata = f.read(dynamic[5])
    except (OSError, struct.error):
        return []

    names: list[str] = []
    for off in range(0, len(dyndata) - 15, 16):
        d_tag, d_val = struct.unpack_from("<qQ", dyndata, off)
        if d_tag == 0:  # DT_NULL terminates the dynamic section
            break
        if d_tag == 1 and d_val < len(strdata):  # DT_NEEDED
            end = strdata.find(b"\x00", d_val)
            if end > d_val:
                names.append(strdata[d_val:end].decode("ascii", errors="replace"))
    return names


def _llamacpp_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    for entry in sys.path:
        lib_dir = Path(entry) / "llama_cpp" / "lib"
        if lib_dir.is_dir():
            key = str(lib_dir.resolve())
            if key not in seen:
                seen.add(key)
                dirs.append(lib_dir)
    return dirs


def required_cuda_sonames() -> list[str]:
    """CUDA sonames the installed llama-cpp-python wheel actually links against.

    Reading DT_NEEDED means we preload exactly the runtime major the wheel was
    built for (cu12 today, cu13 wheels when they ship) on any NVIDIA machine,
    instead of guessing a hardcoded version.
    """
    found: list[str] = []
    for lib_dir in _llamacpp_lib_dirs():
        for so in sorted(lib_dir.glob("*.so")):
            for name in _elf_needed_sonames(so):
                if name.startswith(_CUDA_SONAME_PREFIXES) and name not in found:
                    found.append(name)
    # libcublas dlopens the matching libcublasLt internally; preload it first.
    for name in list(found):
        if name.startswith("libcublas.so."):
            lt_name = name.replace("libcublas.so.", "libcublasLt.so.")
            if lt_name not in found:
                found.insert(found.index(name), lt_name)
    return found


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
    # Editable installs / venv: also scan sys.prefix site-packages.
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


def preload_cuda_shared_libraries(*, lib_dirs: list[str] | None = None) -> list[str]:
    """Load pip-shipped CUDA runtime libs before ``llama_cpp`` import (Linux dlopen quirk).

    ``LD_LIBRARY_PATH`` changes cannot affect the current process, so the CUDA
    runtime libs the llama.cpp wheel links against must already be mapped when
    ``llama_cpp`` is imported. Preloads the exact sonames the installed wheel
    needs (via DT_NEEDED); falls back to one version per library family.
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        return []
    dirs = lib_dirs if lib_dirs is not None else pip_nvidia_cuda_lib_dirs()
    loaded: list[str] = []

    def _load(lib_name: str) -> bool:
        for lib_dir in dirs:
            candidate = Path(lib_dir) / lib_name
            if not candidate.is_file():
                continue
            key = str(candidate.resolve())
            if key in loaded:
                return True
            try:
                ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
                loaded.append(key)
                return True
            except OSError:
                continue
        return False

    required = required_cuda_sonames()
    if required:
        for soname in required:
            _load(soname)
    else:
        # llama_cpp missing or CPU-only: preload one runtime major per family
        # (cu12 preferred) so a CUDA wheel installed later still imports.
        for family in _CUDA_PRELOAD_FAMILIES:
            for version in _CUDA_PRELOAD_VERSIONS:
                if _load(f"{family}.so.{version}"):
                    break
    _cuda_preloaded = bool(loaded)
    return loaded


def ensure_cuda_library_path() -> list[str]:
    """Prepend pip NVIDIA CUDA libs to ``LD_LIBRARY_PATH`` and preload key shared objects."""
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


def _resolve_hardware_profile(profile: dict | None) -> dict | None:
    if profile is not None:
        return profile
    try:
        from seiso.hardware.profile import hardware_profile

        return hardware_profile()
    except ImportError:
        return None


def llamacpp_deferred_preflight_platform(*, profile: dict | None = None) -> str | None:
    """Platform id when llama.cpp load ladder should defer strict preflight blocking.

    Returns ``apple_unified`` or ``linux_nvidia`` when chat loads may succeed via
    mmap, partial GPU offload, and OOM tier recovery even though static fit math
    blocks on current free memory. Returns ``None`` when preflight should stand.
    """
    profile = _resolve_hardware_profile(profile)
    if profile is None:
        return None

    try:
        from seiso.hardware.tiers import HardwareTier, classify_tier

        if classify_tier(profile) == HardwareTier.APPLE_UNIFIED:
            return "apple_unified"
    except ImportError:
        pass

    if is_native_linux_nvidia(profile=profile):
        return "linux_nvidia"
    return None


def is_native_linux_nvidia(*, profile: dict | None = None) -> bool:
    """True on bare-metal Linux with a discrete NVIDIA GPU (not WSL, not CPU-only)."""
    import platform as _platform

    if _platform.system() != "Linux" or detect_wsl2():
        return False
    profile = _resolve_hardware_profile(profile)
    if profile is None:
        return False
    gpus = profile.get("gpus") or []
    if not gpus:
        return False
    try:
        from seiso.hardware.tiers import HardwareTier, classify_tier

        tier = classify_tier(profile)
        if tier in (HardwareTier.APPLE_UNIFIED, HardwareTier.CPU_ONLY):
            return False
    except ImportError:
        pass
    vendor = str(profile.get("vendor") or "").lower()
    if vendor == "nvidia":
        return True
    for gpu in gpus:
        name = str(gpu.get("name") or "").lower()
        if "nvidia" in name or "geforce" in name or "rtx" in name or "quadro" in name:
            return True
    return False
