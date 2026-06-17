"""Secure-boundary policy for NVIDIA GPU training on Linux / WSL2.

CUDA and WSL2 paths compile and run native fused kernels; host venv training
increases driver and JIT attack surface. Call :func:`enforce_nvidia_secure_boundary`
before GPU training unless an approved isolation tier is active.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

_ACK_HOST_VENV_ENV = "SEISO_NVIDIA_HOST_VENV_ACK"
_ACK_SECURE_VM_ENV = "SEISO_NVIDIA_SECURE_VM"
_ACK_WSL_ENV = "SEISO_NVIDIA_WSL_ACK"
_SKIP_BOUNDARY_ENV = "SEISO_SKIP_NVIDIA_BOUNDARY"
_ABORT_ENV = "SEISO_ABORT_ON_SECURITY_BYPASS"

# Backward compatibility with adaptive-rl-quant env vars
_LEGACY_ACK = {
    _ACK_HOST_VENV_ENV: "ADAPTIVE_RL_NVIDIA_HOST_VENV_ACK",
    _ACK_SECURE_VM_ENV: "ADAPTIVE_RL_NVIDIA_SECURE_VM",
    _ACK_WSL_ENV: "ADAPTIVE_RL_NVIDIA_WSL_ACK",
    _SKIP_BOUNDARY_ENV: "ADAPTIVE_RL_SKIP_NVIDIA_BOUNDARY",
    _ABORT_ENV: "ADAPTIVE_RL_ABORT_ON_SECURITY_BYPASS",
}

_NVIDIA_SMI_PATHS = (
    "/usr/bin/nvidia-smi",
    "/usr/lib/nvidia/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
    "/usr/local/cuda/bin/nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",
)

_BOUNDARY_DOC = "docs/platforms/linux-nvidia.md"


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    legacy = _LEGACY_ACK.get(name)
    if legacy:
        raw = os.environ.get(legacy, "").strip().lower()
        return raw in {"1", "true", "yes", "on"}
    return False


def _resolve_nvidia_smi() -> str | None:
    for env_name in ("SEISO_NVIDIA_SMI_PATH", "NVIDIA_SMI_PATH", "ADAPTIVE_RL_NVIDIA_SMI_PATH"):
        raw = os.environ.get(env_name, "").strip()
        if raw and Path(raw).is_file():
            return raw
    which = shutil.which("nvidia-smi")
    if which:
        return which
    for path in _NVIDIA_SMI_PATHS:
        if Path(path).is_file():
            return path
    return None


def nvidia_smi_visible() -> bool:
    """Return True when nvidia-smi reports at least one GPU."""
    exe = _resolve_nvidia_smi()
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def is_linux_nvidia_host() -> bool:
    if platform.system().lower() != "linux":
        return False
    return nvidia_smi_visible()


def in_container() -> bool:
    return Path("/.dockerenv").is_file()


def in_ci() -> bool:
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        return True
    return _env_enabled("CI")


def detect_wsl2() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl2" in version


def recommended_gpu_install_ack_env() -> str:
    if detect_wsl2():
        return _ACK_WSL_ENV
    return _ACK_HOST_VENV_ENV


def approved_nvidia_boundary() -> tuple[str, str] | None:
    if in_container():
        return ("docker_container", "hardened container")
    if _env_enabled(_ACK_SECURE_VM_ENV):
        return ("disposable_vm", _ACK_SECURE_VM_ENV)
    if detect_wsl2() and _env_enabled(_ACK_WSL_ENV):
        return ("wsl2", _ACK_WSL_ENV)
    if _env_enabled(_ACK_HOST_VENV_ENV):
        return ("host_venv", _ACK_HOST_VENV_ENV)
    return None


def nvidia_boundary_report() -> dict[str, object]:
    approved = approved_nvidia_boundary()
    return {
        "linux_nvidia_host": is_linux_nvidia_host(),
        "wsl2": detect_wsl2(),
        "in_container": in_container(),
        "in_ci": in_ci(),
        "approved_tier": approved[0] if approved else None,
        "approved_via": approved[1] if approved else None,
        "skip_boundary": _env_enabled(_SKIP_BOUNDARY_ENV),
    }


def _boundary_failure_message(*, context: str) -> str:
    return (
        f"NVIDIA secure boundary required before {context}.\n"
        "An NVIDIA GPU was detected on Linux. Native CUDA kernel JIT and driver "
        "access increase attack surface on bare host venvs.\n\n"
        "Approve exactly one isolation tier, then retry:\n"
        f"  export {_ACK_SECURE_VM_ENV}=1     # disposable VM / lab host\n"
        f"  export {_ACK_WSL_ENV}=1           # WSL2 (recommended on Windows)\n"
        f"  export {_ACK_HOST_VENV_ENV}=1     # trusted host venv only\n\n"
        f"See {_BOUNDARY_DOC} for setup guidance."
    )


def enforce_nvidia_secure_boundary(*, context: str = "training") -> dict[str, object]:
    """
    Gate GPU training on NVIDIA Linux hosts.

    CI and non-NVIDIA hosts are no-ops. Honors ``SEISO_SKIP_NVIDIA_BOUNDARY`` with
    optional abort via ``SEISO_ABORT_ON_SECURITY_BYPASS``.
    """
    report = nvidia_boundary_report()
    if not report["linux_nvidia_host"] or report["in_ci"]:
        return report

    if _env_enabled(_SKIP_BOUNDARY_ENV):
        message = f"NVIDIA secure boundary skipped during {context} ({_SKIP_BOUNDARY_ENV}=1)."
        if _env_enabled(_ABORT_ENV):
            raise SystemExit(message)
        print(message, file=sys.stderr)
        report["boundary_enforced"] = False
        report["boundary_skipped"] = True
        return report

    approved = approved_nvidia_boundary()
    if approved is not None:
        tier, via = approved
        print(
            f"seiso_nvidia_boundary: ok ({tier} via {via}) — context={context}",
            file=sys.stderr,
        )
        report["boundary_enforced"] = True
        report["approved_tier"] = tier
        report["approved_via"] = via
        return report

    raise SystemExit(_boundary_failure_message(context=context))


__all__ = [
    "approved_nvidia_boundary",
    "detect_wsl2",
    "enforce_nvidia_secure_boundary",
    "in_ci",
    "in_container",
    "is_linux_nvidia_host",
    "nvidia_boundary_report",
    "nvidia_smi_visible",
    "recommended_gpu_install_ack_env",
]
