"""RL quant preset metadata — single source for API, CLI, and UI."""

from __future__ import annotations

from typing import Any

RL_QUANT_PRESETS: list[dict[str, str]] = [
    {
        "id": "reproducible",
        "label": "Reproducible research (simulator)",
        "backend": "simulator",
        "training_backend": "stdlib",
    },
    {
        "id": "minimal",
        "label": "Fast smoke (256 episodes)",
        "backend": "simulator",
        "training_backend": "stdlib",
    },
    {
        "id": "post_train",
        "label": "Post fine-tune RL (continuous, router)",
        "backend": "simulator",
        "training_backend": "stdlib",
    },
]

RL_QUANT_PRESET_HINTS: dict[str, str] = {
    "minimal": "Fast smoke run — simulator backend, few episodes.",
    "reproducible": "Fixed seeds and logged artifacts for paper-grade reproducibility.",
    "post_train": "Post fine-tune checkpoint — links training output to quant recommendation.",
}

REWARD_WEIGHTS_HELP: dict[str, str] = {
    "alpha_latency": "Latency penalty weight",
    "beta_throughput": "Throughput reward weight",
    "gamma_perplexity": "Quality / perplexity weight",
    "delta_memory": "Memory footprint weight",
    "epsilon_instability": "Instability probe penalty",
    "theta_kernel_speedup": "CUDA kernel speedup reward",
    "iota_kernel_latency": "Kernel micro-benchmark latency penalty",
}

KERNEL_RL_HELP: dict[str, Any] = {
    "kernel_rl_enabled": "Co-train quantization policy with CUDA launch profiles (stripe/parallax/SwiGLU vec)",
    "kernel_live_benchmark": "Run cached live GPU micro-benchmarks (NVIDIA CUDA only; slower)",
    "kernel_hidden_dim": "Hidden dimension for kernel bench shapes (default 4096)",
    "kernel_batch_rows": "Token rows for kernel bench shapes (default 4096)",
    "profiles": ["auto", "stripe", "parallax", "narrow_opt", "wide_throughput", "balanced"],
}


AUTO_SWEEP_HELP: dict[str, str] = {
    "auto_sweep": "Run a hyperparameter grid search before the full pipeline (default: on)",
    "sweep_config": "Optional sweep grid JSON/TOML; omit to use preset defaults",
    "sweep_objective": "Metric to rank trials (default: evaluation.mean_reward)",
}


def rl_quant_presets_response() -> dict[str, Any]:
    return {
        "presets": RL_QUANT_PRESETS,
        "preset_hints": RL_QUANT_PRESET_HINTS,
        "reward_weights_help": REWARD_WEIGHTS_HELP,
        "kernel_rl_help": KERNEL_RL_HELP,
        "auto_sweep_help": AUTO_SWEEP_HELP,
    }
