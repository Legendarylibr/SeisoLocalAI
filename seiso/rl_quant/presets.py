"""RL quant product presets — single registry for API, CLI, UI, and sweep grids.

Adaptive-quant ``named_preset`` remains the research FrameworkConfig factory;
this module is the Seiso product contract (ids, aliases, defaults, stages, sweeps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STAGE_ORDER = (
    "auto_sweep",
    "train",
    "evaluate",
    "recommend",
    "benchmark",
    "analysis",
    "paper_bundle",
)

STAGE_HELP: dict[str, str] = {
    "auto_sweep": "Optional hyperparameter sweep before the main policy run",
    "train": "Train the adaptive quantization policy",
    "evaluate": "Evaluate reward and quality tradeoffs",
    "recommend": "Write evidence-tagged quantization recommendations",
    "benchmark": "Run simulator or llama.cpp benchmark suite",
    "analysis": "Generate analysis artifacts from logs and traces",
    "paper_bundle": "Package reproducibility metadata and artifacts",
}

_FULL_STAGES = (
    "auto_sweep",
    "train",
    "evaluate",
    "recommend",
    "benchmark",
    "analysis",
    "paper_bundle",
)

_MINIMAL_STAGES = ("train", "evaluate", "recommend", "benchmark")

_FALLBACK_SWEEP_GRID: dict[str, tuple[Any, ...]] = {
    "learning_rate": (0.02, 0.035),
    "value_learning_rate": (0.015, 0.025),
}


@dataclass(frozen=True)
class RLQuantPreset:
    """Product-facing RL quant preset."""

    id: str
    label: str
    hint: str
    backend: str
    training_backend: str
    stages: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    #: Key passed to ``adaptive_quant.easy_config.named_preset`` (defaults to ``id``).
    named_preset: str | None = None
    #: Bundle-relative prompt library applied for this product preset.
    prompt_library: str | None = None
    sweep_grid: dict[str, tuple[Any, ...]] | None = None

    def resolve_named_preset(self) -> str:
        return self.named_preset or self.id


RL_QUANT_PRESET_DEFS: tuple[RLQuantPreset, ...] = (
    RLQuantPreset(
        id="reproducible",
        label="Reproducible research (simulator)",
        hint="Fixed seeds and logged artifacts for paper-grade reproducibility.",
        backend="simulator",
        training_backend="python",
        stages=_FULL_STAGES,
        aliases=("repro", "reproducible_research"),
        sweep_grid={
            "learning_rate": (0.02, 0.035),
            "value_learning_rate": (0.015, 0.025),
        },
    ),
    RLQuantPreset(
        id="minimal",
        label="Fast smoke (256 episodes)",
        hint="Fast smoke run — simulator backend, few episodes.",
        backend="simulator",
        training_backend="python",
        stages=_MINIMAL_STAGES,
        aliases=("fast", "smoke"),
        sweep_grid={
            "learning_rate": (0.025, 0.035),
        },
    ),
    RLQuantPreset(
        id="post_train",
        label="Post fine-tune RL (continuous, router)",
        hint="Post fine-tune checkpoint — links training output to quant recommendation.",
        backend="simulator",
        training_backend="python",
        stages=_FULL_STAGES,
        aliases=("posttrain", "llm_post_train"),
        prompt_library="prompts/post_train_library.json",
        sweep_grid={
            "learning_rate": (0.02, 0.035),
            "value_learning_rate": (0.015, 0.025),
            "reward_weights.beta_throughput": (0.04, 0.08),
        },
    ),
)

_PRESETS_BY_ID: dict[str, RLQuantPreset] = {p.id: p for p in RL_QUANT_PRESET_DEFS}
_PRESET_ALIASES: dict[str, str] = {
    alias: preset.id
    for preset in RL_QUANT_PRESET_DEFS
    for alias in (preset.id, *preset.aliases)
}

# Back-compat list/dict shapes used by API/UI clients.
RL_QUANT_PRESETS: list[dict[str, Any]] = [
    {
        "id": p.id,
        "label": p.label,
        "backend": p.backend,
        "training_backend": p.training_backend,
        "stages": list(p.stages),
    }
    for p in RL_QUANT_PRESET_DEFS
]

RL_QUANT_PRESET_HINTS: dict[str, str] = {p.id: p.hint for p in RL_QUANT_PRESET_DEFS}

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
    "profiles": [
        "auto",
        "stripe",
        "parallax",
        "narrow_opt",
        "wide_throughput",
        "balanced",
    ],
}


AUTO_SWEEP_HELP: dict[str, str] = {
    "auto_sweep": "Run a hyperparameter grid search before the full pipeline (default: on)",
    "sweep_config": "Optional sweep grid JSON/TOML; omit to use preset defaults",
    "sweep_objective": "Metric to rank trials (default: evaluation.mean_reward)",
}


def _normalize_raw(name: str | None, *, default: str = "reproducible") -> str:
    raw = (name or default).strip().lower().replace("-", "_")
    return raw or default


def lookup_preset(name: str | None) -> RLQuantPreset | None:
    """Return a product preset when ``name`` is a known id/alias; else ``None``."""
    if name is None:
        return None
    raw = _normalize_raw(name)
    canonical = _PRESET_ALIASES.get(raw)
    if canonical is None:
        return None
    return _PRESETS_BY_ID[canonical]


def normalize_preset_id(name: str | None, *, default: str = "reproducible") -> str:
    """Map preset id/alias to the canonical product preset id."""
    preset = lookup_preset(default if name in (None, "") else name)
    if preset is None:
        raise ValueError(
            f"Unknown RL quant preset {name!r}. "
            f"Use: {', '.join(p.id for p in RL_QUANT_PRESET_DEFS)}."
        )
    return preset.id


def get_preset(name: str | None, *, default: str = "reproducible") -> RLQuantPreset:
    """Return the product preset for ``name`` (id or alias)."""
    return _PRESETS_BY_ID[normalize_preset_id(name, default=default)]


def known_preset_ids() -> tuple[str, ...]:
    return tuple(p.id for p in RL_QUANT_PRESET_DEFS)


def sweep_grid_for_preset(name: str | None) -> dict[str, tuple[Any, ...]]:
    """Default auto-sweep grid for a product preset (or fallback if unknown)."""
    preset = lookup_preset(name)
    if preset is None:
        return dict(_FALLBACK_SWEEP_GRID)
    return dict(preset.sweep_grid or _FALLBACK_SWEEP_GRID)


def rl_quant_presets_response() -> dict[str, Any]:
    return {
        "presets": RL_QUANT_PRESETS,
        "stages": list(STAGE_ORDER),
        "help": STAGE_HELP,
        "defaults": {
            "preset": "minimal",
            "training_episodes": 256,
            "evaluation_episodes": 64,
            "backend": "simulator",
            "training_backend": "python",
        },
        "preset_hints": RL_QUANT_PRESET_HINTS,
        "reward_weights_help": REWARD_WEIGHTS_HELP,
        "kernel_rl_help": KERNEL_RL_HELP,
        "auto_sweep_help": AUTO_SWEEP_HELP,
    }
