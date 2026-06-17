from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from adaptive_quant.nvidia_secure_boundary import (
    _ACK_HOST_VENV_ENV,
    _ACK_SECURE_VM_ENV,
    _ACK_WSL_ENV,
    recommended_gpu_install_ack_env,
)
from adaptive_quant.ui.rl_fields import apply_rl_cli_flags, collect_rl_set_overrides


def _cli_path(repo: Path, name: str) -> Path | None:
    scripts = repo / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    if os.name == "nt":
        for candidate in (scripts / f"{name}.exe", scripts / name):
            if candidate.is_file():
                return candidate
        return None
    candidate = scripts / name
    return candidate if candidate.is_file() else None


def _append_common_overrides(command: list[str], options: dict[str, Any]) -> None:
    mapping = (
        ("training_episodes", "--training-episodes"),
        ("evaluation_episodes", "--evaluation-episodes"),
        ("benchmark_training_episodes", "--benchmark-training-episodes"),
        ("benchmark_evaluation_episodes", "--benchmark-evaluation-episodes"),
        ("run_name", "--run-name"),
        ("seed", "--seed"),
    )
    for key, flag in mapping:
        value = options.get(key)
        if value is not None and str(value).strip() != "":
            command.extend([flag, str(value)])

    config = options.get("config")
    if config and str(config).strip():
        command.extend(["--config", str(config).strip()])

    overrides = options.get("config_overrides") or []
    if isinstance(overrides, str):
        overrides = [line.strip() for line in overrides.splitlines() if line.strip()]
    structured = collect_rl_set_overrides(options)
    for item in [*structured, *overrides]:
        command.extend(["--set", str(item).strip()])

    apply_rl_cli_flags(command, options)


def _workflow_command(
    repo: Path,
    python_bin: str,
    console_name: str,
    module_name: str,
) -> list[str]:
    cli = _cli_path(repo, console_name)
    if cli:
        return [str(cli)]
    return [python_bin, "-m", f"adaptive_quant.cli.{module_name}"]


def build_job_env(
    options: dict[str, Any],
    *,
    base_env: dict[str, str] | None = None,
    repo: Path | None = None,
) -> dict[str, str]:
    """Merge UI environment options into a subprocess env dict."""
    env = dict(base_env or os.environ)
    if repo is not None:
        src = repo / "src"
        if src.is_dir():
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(src) + (os.pathsep + existing if existing else "")
    ack = str(options.get("nvidia_ack", "")).strip()
    if ack == "host_venv":
        env[_ACK_HOST_VENV_ENV] = "1"
    elif ack == "wsl":
        env[_ACK_WSL_ENV] = "1"
    elif ack == "vm":
        env[_ACK_SECURE_VM_ENV] = "1"
    elif options.get("accept_gpu_install"):
        env[recommended_gpu_install_ack_env()] = "1"

    smi_path = str(options.get("nvidia_smi_path", "")).strip()
    if smi_path:
        env["ADAPTIVE_RL_NVIDIA_SMI_PATH"] = smi_path

    if options.get("privileged_overrides"):
        env["ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES"] = "1"
    return env


def build_workflow_command(
    *,
    workflow: str,
    options: dict[str, Any] | None,
    repo: Path,
    python_bin: str,
) -> tuple[str, list[str]]:
    """Return (label, argv) for a configured workflow run."""
    opts = dict(options or {})
    workflow = workflow.strip()

    if workflow == "research":
        command = _workflow_command(repo, python_bin, "adaptive-rl-quant", "research")
        _append_common_overrides(command, opts)
        if str(opts.get("config", "")).endswith("config.e2e_smoke.json"):
            label = "E2E smoke run"
        else:
            label = str(opts.get("run_name") or "Research pipeline")
        return (label, command)

    if workflow == "moe":
        command = _workflow_command(repo, python_bin, "adaptive-rl-quant-moe", "moe_research")
        _append_common_overrides(command, opts)
        return ("MoE research", command)

    if workflow == "pytorch":
        command = _workflow_command(repo, python_bin, "adaptive-rl-quant-pytorch", "pytorch")
        preset = str(opts.get("preset") or "gpu").strip()
        if opts.get("config"):
            _append_common_overrides(command, opts)
        else:
            command.extend(["--preset", preset])
            _append_common_overrides(command, {k: v for k, v in opts.items() if k != "preset"})
        return (f"PyTorch preset {preset}", command)

    if workflow == "multiseed":
        command = _workflow_command(repo, python_bin, "adaptive-rl-quant-multiseed", "multiseed")
        command.extend(["--preset", str(opts.get("preset") or "dense")])
        if opts.get("seeds"):
            command.extend(["--seeds", str(opts["seeds"])])
        if opts.get("episodes") is not None:
            command.extend(["--episodes", str(opts["episodes"])])
        if opts.get("run_name"):
            command.extend(["--run-name", str(opts["run_name"])])
        return ("Multiseed experiment", command)

    if workflow == "sweep":
        command = _workflow_command(repo, python_bin, "adaptive-rl-quant-sweep", "sweep")
        if opts.get("sweep_config"):
            command.extend(["--sweep-config", str(opts["sweep_config"])])
        if opts.get("config"):
            command.extend(["--config", str(opts["config"])])
        vary = opts.get("vary") or opts.get("config_overrides")
        lines: list[str]
        if isinstance(vary, str):
            lines = [line.strip() for line in vary.splitlines() if line.strip()]
        elif isinstance(vary, list):
            lines = [str(item).strip() for item in vary if str(item).strip()]
        else:
            lines = []
        for item in lines:
            command.extend(["--vary", item])
        if opts.get("episodes") is not None:
            command.extend(["--episodes", str(opts["episodes"])])
        if opts.get("run_name"):
            command.extend(["--run-name", str(opts["run_name"])])
        return ("Hyperparameter sweep", command)

    if workflow == "online":
        command = _workflow_command(repo, python_bin, "adaptive-rl-quant-online", "online_learning")
        _append_common_overrides(command, opts)
        if opts.get("requests") is not None:
            command.extend(["--requests", str(opts["requests"])])
        return ("Online learning", command)

    if workflow == "continuous":
        command = _workflow_command(
            repo, python_bin, "adaptive-rl-quant-continuous", "continuous_learning"
        )
        _append_common_overrides(command, opts)
        return ("Continuous learning", command)

    if workflow == "frontier":
        command = _workflow_command(repo, python_bin, "adaptive-rl-quant-frontier", "frontier")
        _append_common_overrides(command, opts)
        return ("Frontier comparison", command)

    if workflow == "analyze":
        cli = _cli_path(repo, "adaptive-rl-quant-analyze")
        command = [str(cli)] if cli else [python_bin, "-m", "analysis"]
        return ("Analyze outputs", command)

    if workflow == "install_cuda":
        command = [python_bin, "scripts/install_cuda_torch.py", "--accept-gpu-install"]
        cuda = str(opts.get("cuda") or "cu130").strip()
        if cuda in {"cu130", "cu126"}:
            command.extend(["--cuda", cuda])
        if opts.get("force_reinstall"):
            command.append("--force-reinstall")
        return ("Install CUDA PyTorch", command)

    if workflow == "cuda_check":
        return (
            "CUDA diagnostics",
            [python_bin, "scripts/install_cuda_torch.py", "--check-only"],
        )

    if workflow == "setup_tests":
        command = [python_bin, "scripts/run_setup_tests.py"]
        if opts.get("full"):
            command.append("--full")
        if opts.get("no_torch"):
            command.append("--no-torch")
        if opts.get("no_nvidia"):
            command.append("--no-nvidia")
        return ("Setup tests", command)

    if workflow == "doctor":
        return ("Environment report", [python_bin, "scripts/env_report.py"])

    # Backward-compatible action aliases
    aliases = {
        "smoke": ("research", {"config": "config.e2e_smoke.json"}),
        "full_run": ("research", {}),
    }
    if workflow in aliases:
        alias_workflow, alias_opts = aliases[workflow]
        merged = {**alias_opts, **opts}
        return build_workflow_command(
            workflow=alias_workflow,
            options=merged,
            repo=repo,
            python_bin=python_bin,
        )

    if workflow.startswith("pytorch:"):
        preset = workflow.split(":", 1)[1]
        merged = {**opts, "preset": preset}
        return build_workflow_command(
            workflow="pytorch",
            options=merged,
            repo=repo,
            python_bin=python_bin,
        )

    raise ValueError(f"Unknown workflow: {workflow}")


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
