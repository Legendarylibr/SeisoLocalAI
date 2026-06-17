from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_quant.nvidia_secure_boundary import (
    detect_wsl2,
    recommended_gpu_install_ack_env,
)
from adaptive_quant.ui.rl_fields import field_groups_for_workflow

_COMMON_RUN_FIELDS: list[dict[str, Any]] = [
    {
        "name": "config",
        "label": "Config file",
        "type": "config_file",
        "help": "JSON or TOML path relative to repo root.",
    },
    {
        "name": "training_episodes",
        "label": "Training episodes",
        "type": "int",
        "min": 1,
        "placeholder": "e.g. 48",
    },
    {
        "name": "evaluation_episodes",
        "label": "Evaluation episodes",
        "type": "int",
        "min": 1,
        "placeholder": "e.g. 12",
    },
    {
        "name": "benchmark_training_episodes",
        "label": "Benchmark training episodes",
        "type": "int",
        "min": 1,
        "placeholder": "optional",
    },
    {
        "name": "benchmark_evaluation_episodes",
        "label": "Benchmark evaluation episodes",
        "type": "int",
        "min": 1,
        "placeholder": "optional",
    },
    {
        "name": "run_name",
        "label": "Run name",
        "type": "text",
        "placeholder": "outputs subdirectory name",
    },
    {
        "name": "seed",
        "label": "Seed",
        "type": "int",
        "placeholder": "random seed",
    },
    {
        "name": "config_overrides",
        "label": "Extra overrides",
        "type": "lines",
        "help": "One KEY=VALUE per line (--set). Example: learning_rate=0.03",
    },
]

_ENVIRONMENT_FIELDS: list[dict[str, Any]] = [
    {
        "name": "nvidia_ack",
        "label": "NVIDIA secure boundary",
        "type": "choice",
        "choices": [
            {"value": "", "label": "No ack (diagnostics only)"},
            {"value": "host_venv", "label": "Tier 4 — host .venv"},
            {"value": "wsl", "label": "Tier 3 — WSL2"},
            {"value": "vm", "label": "Tier 1–2 — disposable VM"},
        ],
        "help": "Required for GPU install/training on Linux + NVIDIA.",
    },
    {
        "name": "nvidia_smi_path",
        "label": "nvidia-smi path",
        "type": "text",
        "placeholder": "/usr/bin/nvidia-smi",
    },
    {
        "name": "privileged_overrides",
        "label": "Allow privileged --set overrides",
        "type": "checkbox",
        "help": "Sets ADAPTIVE_RL_ALLOW_PRIVILEGED_OVERRIDES=1 for backend/router paths.",
    },
]

_WORKFLOW_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "research",
        "title": "Research pipeline",
        "description": "Train → evaluate → benchmarks → analysis (simulator or llama.cpp).",
        "category": "run",
        "cli": "adaptive-rl-quant",
        "fields": list(_COMMON_RUN_FIELDS),
        "presets": [
            {"label": "E2E smoke", "options": {"config": "config.e2e_smoke.json"}},
            {
                "label": "Smoke + diversity prompts",
                "options": {
                    "config": "config.e2e_smoke.json",
                    "prompt_library_path": "prompts/diversity_library.json",
                    "env_sampling_mode": "sequential",
                    "seed": 13,
                },
            },
            {
                "label": "Fast prompt loop",
                "options": {
                    "config": "config.e2e_smoke.json",
                    "prompt_library_path": "prompts/smoke_library.json",
                    "training_episodes": 16,
                    "evaluation_episodes": 4,
                    "env_sampling_mode": "sequential",
                    "write_research_report": True,
                },
            },
            {"label": "Full baseline", "options": {}},
        ],
    },
    {
        "id": "moe",
        "title": "MoE research",
        "description": "Mixture-of-experts quantization research pipeline.",
        "category": "run",
        "cli": "adaptive-rl-quant-moe",
        "fields": list(_COMMON_RUN_FIELDS),
    },
    {
        "id": "pytorch",
        "title": "PyTorch / CUDA training",
        "description": "GPU trainer with PPO/VPG-style updates.",
        "category": "gpu",
        "cli": "adaptive-rl-quant-pytorch",
        "requires_nvidia_ack": True,
        "fields": [
            {
                "name": "preset",
                "label": "Preset",
                "type": "choice",
                "choices": [
                    {"value": "gpu", "label": "gpu — auto VRAM profile"},
                    {"value": "3090", "label": "3090 — RTX 3090 host"},
                    {"value": "4090", "label": "4090 — RTX 4090 host"},
                    {"value": "4090-universal", "label": "4090-universal — multi-hardware"},
                    {"value": "post-train", "label": "post-train — long routed RL"},
                ],
                "default": "gpu",
            },
            *_COMMON_RUN_FIELDS,
        ],
    },
    {
        "id": "multiseed",
        "title": "Multiseed experiment",
        "description": "Run a preset across multiple seeds and aggregate metrics.",
        "category": "experiment",
        "cli": "adaptive-rl-quant-multiseed",
        "fields": [
            {
                "name": "preset",
                "label": "Preset",
                "type": "choice",
                "choices": [
                    {"value": "dense", "label": "dense"},
                    {"value": "moe", "label": "moe"},
                ],
                "default": "dense",
            },
            {
                "name": "seeds",
                "label": "Seeds",
                "type": "text",
                "default": "13,17,23",
                "placeholder": "13,17,23 or 0-4",
            },
            {
                "name": "episodes",
                "label": "Training episodes",
                "type": "int",
                "min": 1,
                "placeholder": "optional short run",
            },
            {"name": "run_name", "label": "Run name", "type": "text"},
        ],
    },
    {
        "id": "sweep",
        "title": "Hyperparameter sweep",
        "description": "Grid search over config fields with ranked trials.",
        "category": "experiment",
        "cli": "adaptive-rl-quant-sweep",
        "fields": [
            {
                "name": "sweep_config",
                "label": "Sweep config",
                "type": "config_file",
                "default": "config.sweep.example.json",
            },
            {
                "name": "config",
                "label": "Base config (optional)",
                "type": "config_file",
            },
            {
                "name": "vary",
                "label": "Vary grid",
                "type": "lines",
                "help": "One per line: learning_rate=0.02,0.035",
                "default": "learning_rate=0.02,0.035",
            },
            {
                "name": "episodes",
                "label": "Training episodes",
                "type": "int",
                "min": 1,
            },
            {"name": "run_name", "label": "Run name", "type": "text"},
        ],
    },
    {
        "id": "online",
        "title": "Online learning",
        "description": "Simulated serving with replay updates and rollback.",
        "category": "run",
        "cli": "adaptive-rl-quant-online",
        "fields": [
            *_COMMON_RUN_FIELDS,
            {
                "name": "requests",
                "label": "Online requests",
                "type": "int",
                "min": 1,
            },
        ],
    },
    {
        "id": "continuous",
        "title": "Continuous learning",
        "description": "Long-running continuous adaptation pipeline.",
        "category": "run",
        "cli": "adaptive-rl-quant-continuous",
        "fields": list(_COMMON_RUN_FIELDS),
        "presets": [
            {
                "label": "Continuous smoke",
                "options": {"config": "config.e2e_continuous_smoke.json"},
            },
            {
                "label": "Reward-adaptive long run",
                "options": {
                    "config": "config.e2e_continuous_smoke.json",
                    "continuous_task_stream_mode": "reward_adaptive",
                    "prompt_library_path": "prompts/diversity_library.json",
                    "continuous_max_tasks": 256,
                    "write_research_report": True,
                },
            },
            {
                "label": "Chat task stream (JSONL)",
                "options": {
                    "config": "config.e2e_continuous_smoke.json",
                    "continuous_task_stream_mode": "jsonl",
                    "continuous_task_jsonl_path": "outputs/chat_tasks.jsonl",
                    "continuous_max_tasks": 128,
                },
            },
        ],
    },
    {
        "id": "frontier",
        "title": "Frontier comparison",
        "description": "Compare adaptive quantization against frontier baselines.",
        "category": "experiment",
        "cli": "adaptive-rl-quant-frontier",
        "fields": list(_COMMON_RUN_FIELDS),
    },
    {
        "id": "analyze",
        "title": "Analyze outputs",
        "description": "Regenerate analysis from existing logs under outputs/.",
        "category": "verify",
        "cli": "adaptive-rl-quant-analyze",
        "fields": [],
    },
    {
        "id": "install_cuda",
        "title": "Install CUDA PyTorch",
        "description": "Install a GPU torch wheel into the repo venv.",
        "category": "gpu",
        "requires_nvidia_ack": True,
        "fields": [
            {
                "name": "cuda",
                "label": "CUDA wheel",
                "type": "choice",
                "choices": [
                    {"value": "cu130", "label": "cu130 (default)"},
                    {"value": "cu126", "label": "cu126 (legacy drivers)"},
                ],
                "default": "cu130",
            },
            {
                "name": "force_reinstall",
                "label": "Force reinstall",
                "type": "checkbox",
            },
        ],
    },
    {
        "id": "cuda_check",
        "title": "CUDA diagnostics",
        "description": "Check torch/CUDA install without changing packages.",
        "category": "verify",
        "fields": [],
    },
    {
        "id": "setup_tests",
        "title": "Setup tests",
        "description": "Hardware-aware unittest subset used during ./setup.sh.",
        "category": "verify",
        "fields": [
            {
                "name": "full",
                "label": "Full unittest suite",
                "type": "checkbox",
            },
            {
                "name": "no_torch",
                "label": "Skip torch modules",
                "type": "checkbox",
            },
            {
                "name": "no_nvidia",
                "label": "Skip NVIDIA modules",
                "type": "checkbox",
            },
        ],
    },
    {
        "id": "doctor",
        "title": "Environment report",
        "description": "Detailed doctor output (make doctor).",
        "category": "verify",
        "fields": [],
    },
]


def list_config_files(repo: Path) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for path in sorted(repo.glob("config*.json")):
        if path.is_file():
            files.append({"path": path.name, "label": path.name})
    for path in sorted(repo.glob("config*.toml")):
        if path.is_file():
            files.append({"path": path.name, "label": path.name})
    return files


def launcher_catalog(*, repo: Path, status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return workflow definitions and selectable options for the dashboard."""
    if status is None:
        from adaptive_quant.ui.status import dashboard_status

        status = dashboard_status(repo=repo)

    nvidia = status.get("nvidia", {})
    recommended_ack = recommended_gpu_install_ack_env() if nvidia.get("linux_nvidia_host") else None

    workflows: list[dict[str, Any]] = []
    for workflow in _WORKFLOW_DEFINITIONS:
        entry = dict(workflow)
        entry["field_groups"] = field_groups_for_workflow(entry["id"])
        if entry.get("requires_nvidia_ack") and nvidia.get("needs_ack_for_gpu_training"):
            entry["nvidia_ack_required"] = True
        workflows.append(entry)

    return {
        "workflows": workflows,
        "config_files": list_config_files(repo),
        "environment_fields": list(_ENVIRONMENT_FIELDS),
        "recommended_nvidia_ack": (
            "host_venv"
            if recommended_ack == "ADAPTIVE_RL_NVIDIA_HOST_VENV_ACK"
            else "wsl"
            if recommended_ack == "ADAPTIVE_RL_NVIDIA_WSL_ACK"
            else None
        ),
        "platform": {
            "wsl2": detect_wsl2(),
            "linux_nvidia_host": nvidia.get("linux_nvidia_host", False),
        },
    }
