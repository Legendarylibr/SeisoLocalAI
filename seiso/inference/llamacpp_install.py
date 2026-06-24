"""Install and verify CUDA-capable llama-cpp-python for GGUF chat."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

_LLAMACPP_SPEC = "llama-cpp-python>=0.3"
_CUDA_WHEEL_INDEXES = (
    "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    "https://abetlen.github.io/llama-cpp-python/whl/cu121",
)


def llamacpp_import_ok() -> tuple[bool, str | None]:
    try:
        from seiso.platform import ensure_cuda_library_path

        ensure_cuda_library_path()
    except ImportError:
        pass
    try:
        import llama_cpp  # noqa: F401

        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def nvidia_hardware_visible() -> bool:
    try:
        from seiso.security.nvidia_boundary import nvidia_smi_visible

        return nvidia_smi_visible()
    except ImportError:
        return False


def llamacpp_gpu_offload_supported() -> bool:
    """True when the installed wheel can offload layers to GPU (CUDA/Metal)."""
    try:
        import llama_cpp

        for candidate in (
            getattr(llama_cpp, "llama_supports_gpu_offload", None),
            getattr(getattr(llama_cpp, "llama_cpp", None), "llama_supports_gpu_offload", None),
        ):
            if callable(candidate):
                return bool(candidate())
    except Exception:
        pass
    return False


def pip_install_strategies(*, prefer_cuda: bool) -> list[list[str]]:
    """Ordered pip attempts — CUDA wheels first when an NVIDIA GPU is visible."""
    base = [sys.executable, "-m", "pip", "install", "-U", _LLAMACPP_SPEC]
    strategies: list[list[str]] = []

    if prefer_cuda:
        for index in _CUDA_WHEEL_INDEXES:
            strategies.append([*base, "--only-binary", ":all:", "--extra-index-url", index])
            strategies.append([*base, "--extra-index-url", index])

    strategies.extend(
        [
            [*base, "--only-binary", ":all:"],
            base,
        ]
    )
    return strategies


def _pip_install_from_source_cuda() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    # Source build requires the CUDA toolkit (nvcc). If it's missing, skip
    # silently — the caller falls back to a CPU wheel.
    nvcc = shutil.which("nvcc")
    if not nvcc:
        logger.debug("Skipping CUDA source build: nvcc not found in PATH")
        return False
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DLLAMA_CUDA=on"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", _LLAMACPP_SPEC, "--no-cache-dir"],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("llama-cpp-python source CUDA build failed: %s", exc)
        return False
    return proc.returncode == 0


def pip_install_llamacpp(*, prefer_cuda: bool | None = None) -> bool:
    if prefer_cuda is None:
        prefer_cuda = nvidia_hardware_visible()

    for cmd in pip_install_strategies(prefer_cuda=prefer_cuda):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("llama-cpp-python install failed: %s", exc)
            continue
        if proc.returncode == 0:
            return True
        logger.debug(
            "llama-cpp-python install failed (%s): %s",
            " ".join(cmd[-6:]),
            (proc.stderr or proc.stdout or "").strip()[:500],
        )

    if prefer_cuda:
        return _pip_install_from_source_cuda()
    return False


def ensure_llamacpp_installed(*, auto_install: bool | None = None) -> dict[str, Any]:
    """
    Ensure ``llama_cpp`` imports and, when NVIDIA hardware is visible, prefer a GPU-capable wheel.
    """
    if auto_install is None:
        auto_install = os.environ.get("SEISO_SKIP_LLAMACPP_INSTALL", "").strip().lower() not in {
            "1",
            "true",
            "yes",
        }

    want_cuda = nvidia_hardware_visible()
    ok, error = llamacpp_import_ok()
    gpu_ok = llamacpp_gpu_offload_supported() if ok else False
    installed = False

    if ok and (not want_cuda or gpu_ok):
        return {
            "llamacpp": True,
            "installed": False,
            "gpu_offload": gpu_ok,
            "error": None,
        }

    if not auto_install:
        if ok and want_cuda and not gpu_ok:
            error = error or "CPU-only llama-cpp-python; GPU offload unavailable"
        return {
            "llamacpp": ok,
            "installed": False,
            "gpu_offload": gpu_ok,
            "error": error,
        }

    if ok and want_cuda and not gpu_ok:
        logger.info("Replacing CPU-only llama-cpp-python with CUDA-capable build")
    elif not ok:
        logger.info("llama-cpp-python missing — attempting install for GGUF chat")

    installed = pip_install_llamacpp(prefer_cuda=want_cuda)
    if installed:
        _clear_runtime_caches()

    ok, error = llamacpp_import_ok()
    gpu_ok = llamacpp_gpu_offload_supported() if ok else False

    if ok and want_cuda and not gpu_ok:
        logger.warning(
            "llama-cpp-python imports but GPU offload is unavailable — "
            "GGUF chat may run on CPU only"
        )

    if not ok:
        logger.warning("GGUF chat unavailable: %s", error or "llama_cpp import failed")

    return {
        "llamacpp": ok,
        "installed": installed,
        "gpu_offload": gpu_ok,
        "error": error,
    }


def _clear_runtime_caches() -> None:
    try:
        from forge.services.hf_connectivity import check_inference_runtime

        check_inference_runtime.cache_clear()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Install/verify llama-cpp-python for Seiso GGUF chat"
    )
    parser.add_argument("--quiet", action="store_true", help="Only exit code, no stdout")
    args = parser.parse_args(argv)

    result = ensure_llamacpp_installed()
    if result["llamacpp"]:
        if not args.quiet:
            if result.get("gpu_offload"):
                print("llama-cpp-python: OK (GPU offload supported)")
            else:
                print("llama-cpp-python: OK")
        return 0
    if not args.quiet:
        print(
            f"llama-cpp-python: FAILED ({result.get('error') or 'import failed'})",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
