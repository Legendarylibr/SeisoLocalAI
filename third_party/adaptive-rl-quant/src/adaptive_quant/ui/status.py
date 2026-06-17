from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from adaptive_quant.ui.catalog import list_config_files


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _detect_wsl2() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl2" in version


def _git(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _output_counts(repo: Path) -> dict[str, int | None]:
    base = repo / "outputs"
    if not base.is_dir():
        return {}
    counts: dict[str, int | None] = {}
    for name in ("benchmarks", "logs", "analysis", "checkpoints", "reports"):
        directory = base / name
        counts[name] = (
            sum(1 for path in directory.rglob("*") if path.is_file())
            if directory.is_dir()
            else None
        )
    return counts


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("adaptive-rl-quant")
    except importlib.metadata.PackageNotFoundError:
        return None


def _venv_python(repo: Path) -> Path | None:
    if os.name == "nt":
        candidate = repo / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _cli_path(repo: Path, name: str) -> str | None:
    if os.name == "nt":
        scripts = repo / ".venv" / "Scripts"
        for candidate in (scripts / f"{name}.exe", scripts / name):
            if candidate.is_file():
                try:
                    return str(candidate.relative_to(repo))
                except ValueError:
                    return str(candidate)
        return None
    candidate = repo / ".venv" / "bin" / name
    if candidate.is_file():
        try:
            return str(candidate.relative_to(repo))
        except ValueError:
            return str(candidate)
    return None


def _setup_ready(repo: Path) -> dict[str, Any]:
    venv_python = _venv_python(repo)
    import_ok = False
    import_error: str | None = None
    try:
        import adaptive_quant  # noqa: F401

        import_ok = True
    except ImportError as exc:
        import_error = str(exc)

    return {
        "venv_exists": (repo / ".venv").is_dir(),
        "venv_python": str(venv_python) if venv_python is not None else None,
        "package_importable": import_ok,
        "import_error": import_error,
        "package_version": _package_version(),
        "cli_research": _cli_path(repo, "adaptive-rl-quant"),
        "cli_pytorch": _cli_path(repo, "adaptive-rl-quant-pytorch"),
        "cli_ui": _cli_path(repo, "adaptive-rl-quant-ui"),
    }


def _torch_status() -> dict[str, Any]:
    from adaptive_quant.torch_install import INSTALL_CUDA_TORCH_SCRIPT, torch_cuda_ready_report

    report = dict(torch_cuda_ready_report())
    report["install_script"] = INSTALL_CUDA_TORCH_SCRIPT
    return report


def _nvidia_status() -> dict[str, Any]:
    from adaptive_quant.nvidia_secure_boundary import (
        approved_nvidia_boundary,
        is_linux_nvidia_host,
        nvidia_boundary_report,
    )

    report = nvidia_boundary_report()
    approved = approved_nvidia_boundary()
    return {
        "linux_nvidia_host": is_linux_nvidia_host(),
        "boundary": report,
        "approved_tier": approved[0] if approved else None,
        "needs_ack_for_gpu_training": bool(
            report.get("linux_nvidia_host") and not report.get("in_ci") and approved is None
        ),
    }


def _rust_status() -> dict[str, Any]:
    try:
        from adaptive_quant.configuration import FrameworkConfig
        from adaptive_quant.rust_cli import rust_cli_status

        return dict(
            rust_cli_status(FrameworkConfig(run_name="ui_status", detect_host_hardware=False))
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _hardware_status() -> dict[str, Any]:
    from adaptive_quant.hardware import detect_host_hardware

    detected = detect_host_hardware()
    return detected.to_metadata()


def dashboard_status(*, repo: Path | None = None) -> dict[str, Any]:
    """Structured environment snapshot for the launcher dashboard."""
    root = repo if repo is not None else _repo_root()
    on_windows_mount = root.resolve().as_posix().startswith("/mnt/")
    is_wsl2 = _detect_wsl2()
    head = _git(root, "rev-parse", "--short", "HEAD")
    dirty = _git(root, "status", "--porcelain")
    setup = _setup_ready(root)

    readiness = "ready"
    if not setup["venv_exists"] or not setup["package_importable"]:
        readiness = "needs_setup"
    elif is_wsl2 and on_windows_mount:
        readiness = "warning"

    return {
        "readiness": readiness,
        "repo_root": str(root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "wsl2": is_wsl2,
            "repo_on_windows_mount": on_windows_mount,
        },
        "setup": setup,
        "hardware": _hardware_status(),
        "torch": _torch_status(),
        "nvidia": _nvidia_status(),
        "rust": _rust_status(),
        "git": {
            "head": head,
            "dirty": bool((dirty or "").strip()),
        },
        "outputs": _output_counts(root),
        "configs": {
            "files": list_config_files(root),
            "smoke": "config.e2e_smoke.json"
            if (root / "config.e2e_smoke.json").is_file()
            else None,
            "example": "config.example.json" if (root / "config.example.json").is_file() else None,
        },
    }
