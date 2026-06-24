"""Secure-boundary policy for NVIDIA GPU training on Linux / WSL2.

CUDA and WSL2 paths compile and run native fused kernels; host venv training
increases driver and JIT attack surface. Call :func:`enforce_nvidia_secure_boundary`
before GPU training unless an approved isolation tier is active.
"""

from __future__ import annotations

import csv
import io
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from seiso.platform import detect_wsl2

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
_QUERY_TTL_S = 30.0
_query_cache: list[dict[str, object]] | None = None
_query_cache_ts: float = 0.0


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    legacy = _LEGACY_ACK.get(name)
    if legacy:
        raw = os.environ.get(legacy, "").strip().lower()
        return raw in {"1", "true", "yes", "on"}
    return False


def resolve_nvidia_smi_executable() -> str | None:
    """Return an executable ``nvidia-smi`` path (env, PATH, then common driver locations)."""
    for env_name in ("SEISO_NVIDIA_SMI_PATH", "NVIDIA_SMI_PATH", "ADAPTIVE_RL_NVIDIA_SMI_PATH"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            candidate = Path(raw)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
    which = shutil.which("nvidia-smi")
    if which:
        return which
    for path in _NVIDIA_SMI_PATHS:
        candidate = Path(path)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return None


def _resolve_nvidia_smi() -> str | None:
    return resolve_nvidia_smi_executable()


def _run_nvidia_smi(
    exe: str, *args: str, timeout: float = 10.0
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_nvidia_smi_csv(
    stdout: str,
    *,
    fields: tuple[str, ...],
) -> list[dict[str, object]]:
    gpus: list[dict[str, object]] = []
    reader = csv.reader(io.StringIO(stdout.strip()))
    for parts in reader:
        if len(parts) < len(fields):
            continue
        record: dict[str, object] = {}
        skip_row = False
        for field_name, raw in zip(fields, parts, strict=False):
            value = raw.strip()
            if field_name == "index":
                try:
                    record["index"] = int(value)
                except (TypeError, ValueError):
                    record["index"] = len(gpus)
            elif field_name == "name":
                record["name"] = value
            elif field_name == "memory.total":
                try:
                    record["memory_total_mb"] = int(float(value))
                except (TypeError, ValueError):
                    skip_row = True
                    break
        if skip_row:
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if "index" not in record:
            record["index"] = len(gpus)
        if "memory_total_mb" not in record:
            record["memory_total_mb"] = None
        gpus.append(record)
    return gpus


def _query_nvidia_gpus_csv(exe: str, query: str) -> list[dict[str, object]]:
    proc = _run_nvidia_smi(
        exe,
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        timeout=3,
    )
    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        return []
    fields = tuple(part.strip() for part in query.split(","))
    return _parse_nvidia_smi_csv(proc.stdout, fields=fields)


def _query_nvidia_gpus_list(exe: str) -> list[dict[str, object]]:
    """Parse ``nvidia-smi -L`` when CSV queries are unavailable."""
    proc = _run_nvidia_smi(exe, "-L", timeout=3)
    if proc is None or proc.returncode != 0:
        return []
    gpus: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text.startswith("GPU "):
            continue
        body = text.removeprefix("GPU ").strip()
        idx_text, _, rest = body.partition(":")
        try:
            index = int(idx_text.strip())
        except ValueError:
            index = len(gpus)
        name = rest.split("(", 1)[0].strip()
        if not name:
            continue
        gpus.append({"index": index, "name": name, "memory_total_mb": None})
    return gpus


def _probe_nvidia_gpus_uncached() -> list[dict[str, object]]:
    exe = _resolve_nvidia_smi()
    if not exe:
        return []
    for query in ("index,name,memory.total", "name,memory.total", "name"):
        gpus = _query_nvidia_gpus_csv(exe, query)
        if gpus:
            return gpus
    gpus = _query_nvidia_gpus_list(exe)
    if not gpus:
        return gpus
    mem_proc = _run_nvidia_smi(
        exe,
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
        timeout=3,
    )
    if mem_proc is None or mem_proc.returncode != 0:
        return gpus
    memory_values: list[int] = []
    for line in mem_proc.stdout.strip().splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            memory_values.append(int(float(raw)))
        except (TypeError, ValueError):
            memory_values.append(0)
    for idx, gpu in enumerate(gpus):
        if idx < len(memory_values) and memory_values[idx] > 0:
            gpu["memory_total_mb"] = memory_values[idx]
    return gpus


def query_nvidia_gpus(*, force_refresh: bool = False) -> list[dict[str, object]]:
    """Enumerate NVIDIA GPUs via nvidia-smi without PyTorch (cached 30s)."""
    global _query_cache, _query_cache_ts

    now = time.time()
    if not force_refresh and _query_cache is not None and now - _query_cache_ts < _QUERY_TTL_S:
        return _query_cache

    gpus = _probe_nvidia_gpus_uncached()
    _query_cache = gpus
    _query_cache_ts = now
    return gpus


def clear_nvidia_gpu_query_cache() -> None:
    """Clear cached nvidia-smi probe results (tests / post-install)."""
    global _query_cache, _query_cache_ts

    _query_cache = None
    _query_cache_ts = 0.0


def nvidia_smi_visible() -> bool:
    """Return True when nvidia-smi reports at least one GPU."""
    return bool(query_nvidia_gpus())


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
    "clear_nvidia_gpu_query_cache",
    "detect_wsl2",
    "enforce_nvidia_secure_boundary",
    "in_ci",
    "in_container",
    "is_linux_nvidia_host",
    "nvidia_boundary_report",
    "nvidia_smi_visible",
    "query_nvidia_gpus",
    "recommended_gpu_install_ack_env",
    "resolve_nvidia_smi_executable",
]
