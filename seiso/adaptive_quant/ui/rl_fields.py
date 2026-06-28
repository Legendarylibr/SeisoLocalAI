from __future__ import annotations

from typing import Any

# Shared workflow sets
_RUN_WORKFLOWS = frozenset(
    {"research", "moe", "pytorch", "online", "continuous", "frontier"}
)
_TRAIN_WORKFLOWS = frozenset({"research", "moe", "pytorch", "frontier"})

_REWARD_WEIGHT_FIELDS: list[dict[str, Any]] = [
    {
        "name": "reward_weights.alpha_latency",
        "label": "Latency (α)",
        "type": "float",
        "min": 0,
        "max": 5,
        "step": 0.001,
        "placeholder": "0.020",
    },
    {
        "name": "reward_weights.beta_throughput",
        "label": "Throughput (β)",
        "type": "float",
        "min": 0,
        "max": 5,
        "step": 0.001,
        "placeholder": "0.060",
    },
    {
        "name": "reward_weights.gamma_perplexity",
        "label": "Perplexity (γ)",
        "type": "float",
        "min": 0,
        "max": 5,
        "step": 0.01,
        "placeholder": "0.850",
    },
    {
        "name": "reward_weights.delta_memory",
        "label": "Memory (δ)",
        "type": "float",
        "min": 0,
        "max": 5,
        "step": 0.001,
        "placeholder": "0.002",
    },
    {
        "name": "reward_weights.epsilon_instability",
        "label": "Instability (ε)",
        "type": "float",
        "min": 0,
        "max": 10,
        "step": 0.01,
        "placeholder": "1.000",
    },
    {
        "name": "reward_weights.eta_token_latency",
        "label": "Token latency (η)",
        "type": "float",
        "min": 0,
        "max": 5,
        "step": 0.001,
        "placeholder": "0.0",
    },
    {
        "name": "reward_weights.zeta_perplexity_over_ref",
        "label": "Perplexity over ref (ζ)",
        "type": "float",
        "min": 0,
        "max": 5,
        "step": 0.01,
        "placeholder": "0.0",
    },
]

RL_FIELD_GROUPS: list[dict[str, Any]] = [
    {
        "id": "episode_budget",
        "title": "Episode budget & scheduling",
        "description": "Training length, benchmark episodes, and checkpoint cadence.",
        "workflows": sorted(_RUN_WORKFLOWS),
        "fields": [
            {
                "name": "max_training_episodes",
                "label": "Max training episodes",
                "type": "int",
                "min": 1,
                "placeholder": "50000",
            },
            {
                "name": "eval_interval",
                "label": "Eval interval (episodes)",
                "type": "int",
                "min": 1,
                "placeholder": "1000",
            },
            {
                "name": "checkpoint_interval",
                "label": "Checkpoint interval (episodes)",
                "type": "int",
                "min": 1,
                "placeholder": "5000",
            },
            {
                "name": "continuous_training",
                "label": "Continuous training mode",
                "type": "checkbox",
                "help": "Keep training across eval/checkpoint boundaries.",
            },
            {
                "name": "recommendation_eval_episodes",
                "label": "Recommendation eval episodes",
                "type": "int",
                "min": 1,
                "placeholder": "96",
            },
            {
                "name": "write_research_report",
                "label": "Write research report",
                "type": "checkbox",
                "help": "Emit markdown report under outputs/ after the pipeline completes.",
            },
        ],
    },
    {
        "id": "python_trainer",
        "title": "Python RL policy trainer",
        "description": "Stdlib policy-gradient learner (research / simulator path).",
        "workflows": sorted(_TRAIN_WORKFLOWS | {"continuous"}),
        "fields": [
            {
                "name": "learning_rate",
                "label": "Policy learning rate",
                "type": "float",
                "min": 1e-6,
                "max": 1.0,
                "step": 0.001,
                "placeholder": "0.035",
            },
            {
                "name": "value_learning_rate",
                "label": "Value head learning rate",
                "type": "float",
                "min": 1e-6,
                "max": 1.0,
                "step": 0.001,
                "placeholder": "0.020",
            },
            {
                "name": "continuous_stddev",
                "label": "Continuous action stddev",
                "type": "float",
                "min": 0.01,
                "max": 2.0,
                "step": 0.01,
                "placeholder": "0.18",
            },
            {
                "name": "rl_train_policy_mode",
                "label": "Policy sampling mode",
                "type": "choice",
                "choices": [
                    {"value": "stochastic", "label": "stochastic"},
                    {"value": "deterministic", "label": "deterministic"},
                ],
                "default": "stochastic",
            },
        ],
    },
    {
        "id": "reward",
        "title": "Reward engineering",
        "description": "Weighted objective terms combined by compute_weighted_reward.",
        "workflows": sorted(_RUN_WORKFLOWS),
        "fields": [
            *_REWARD_WEIGHT_FIELDS,
            {
                "name": "reward_perplexity_reference",
                "label": "Perplexity reference",
                "type": "float",
                "min": 0,
                "step": 0.1,
                "placeholder": "optional baseline PPL",
            },
        ],
    },
    {
        "id": "safety",
        "title": "Stability & guardrails",
        "description": "Instability probes and safe fallback bit width.",
        "workflows": sorted(_TRAIN_WORKFLOWS | {"continuous"}),
        "fields": [
            {
                "name": "stability_probe_count",
                "label": "Stability probe count",
                "type": "int",
                "min": 0,
                "placeholder": "3",
            },
            {
                "name": "instability_threshold",
                "label": "Instability threshold",
                "type": "float",
                "min": 0,
                "step": 0.1,
                "placeholder": "2.5",
            },
            {
                "name": "safe_default_bits",
                "label": "Safe default bits",
                "type": "int",
                "min": 1,
                "max": 8,
                "placeholder": "4",
            },
        ],
    },
    {
        "id": "environment",
        "title": "Environment sampling",
        "description": "How prompts and hardware profiles are drawn each episode.",
        "workflows": sorted(_TRAIN_WORKFLOWS | {"continuous"}),
        "fields": [
            {
                "name": "prompt_library_path",
                "label": "Prompt library",
                "type": "choice",
                "choices": [
                    {"value": "", "label": "Built-in default (12 prompts)"},
                    {
                        "value": "prompts/smoke_library.json",
                        "label": "Smoke — 4 fast prompts",
                    },
                    {
                        "value": "prompts/diversity_library.json",
                        "label": "Diversity — 8 domains",
                    },
                    {
                        "value": "prompts/post_train_library.json",
                        "label": "Post-train — full curriculum",
                    },
                ],
                "default": "",
                "help": "JSON test-prompt curriculum for training and evaluation episodes.",
            },
            {
                "name": "env_sampling_mode",
                "label": "Sampling mode",
                "type": "choice",
                "choices": [
                    {"value": "random", "label": "random"},
                    {"value": "sequential", "label": "sequential"},
                    {"value": "forced", "label": "forced"},
                ],
                "default": "random",
            },
            {
                "name": "env_forced_prompt_id",
                "label": "Forced prompt id",
                "type": "text",
                "placeholder": "when mode=forced",
            },
            {
                "name": "env_forced_hardware",
                "label": "Forced hardware profile",
                "type": "text",
                "placeholder": "gpu | cpu | low_resource",
            },
            {
                "name": "stability_probe_sampling",
                "label": "Probe sampling",
                "type": "choice",
                "choices": [
                    {"value": "random", "label": "random"},
                    {"value": "deterministic", "label": "deterministic"},
                ],
                "default": "random",
            },
            {
                "name": "prompt_split_enabled",
                "label": "Enable prompt train/eval split",
                "type": "checkbox",
            },
            {
                "name": "prompt_train_fraction",
                "label": "Train fraction",
                "type": "float",
                "min": 0.1,
                "max": 0.99,
                "step": 0.05,
                "placeholder": "0.8",
            },
        ],
    },
    {
        "id": "adaptive",
        "title": "Adaptive quantization behavior",
        "description": "What the policy is allowed to change during RL.",
        "workflows": sorted(_TRAIN_WORKFLOWS | {"moe"}),
        "fields": [
            {
                "name": "multi_hardware",
                "label": "Multi-hardware training",
                "type": "checkbox",
            },
            {
                "name": "dynamic_quant",
                "label": "Dynamic quantization schedules",
                "type": "checkbox",
            },
            {
                "name": "learned_quant",
                "label": "Learned continuous quant knobs",
                "type": "checkbox",
            },
            {
                "name": "detect_host_hardware",
                "label": "Detect host hardware",
                "type": "checkbox",
            },
            {
                "name": "quant_mode",
                "label": "Quantization mode",
                "type": "choice",
                "choices": [
                    {"value": "adaptive", "label": "adaptive"},
                    {"value": "static", "label": "static"},
                ],
                "default": "adaptive",
            },
            {
                "name": "num_groups",
                "label": "Quant groups",
                "type": "int",
                "min": 1,
                "placeholder": "layer groups",
            },
            {
                "name": "num_layers",
                "label": "Simulated layers",
                "type": "int",
                "min": 1,
                "placeholder": "for simulator",
            },
        ],
    },
    {
        "id": "pytorch",
        "title": "PyTorch / CUDA trainer",
        "description": "Neural policy and GPU trainer hyperparameters.",
        "workflows": ["pytorch"],
        "fields": [
            {
                "name": "torch_policy_algorithm",
                "label": "Policy algorithm",
                "type": "choice",
                "choices": [
                    {"value": "ppo", "label": "PPO"},
                    {"value": "vpg", "label": "VPG"},
                    {"value": "awr", "label": "AWR"},
                ],
                "default": "ppo",
            },
            {
                "name": "torch_learning_rate",
                "label": "Torch learning rate",
                "type": "float",
                "min": 1e-7,
                "max": 0.1,
                "step": 0.0001,
                "placeholder": "0.0003",
            },
            {
                "name": "torch_batch_episodes",
                "label": "Batch episodes",
                "type": "int",
                "min": 1,
                "placeholder": "256",
            },
            {
                "name": "torch_minibatch_size",
                "label": "Minibatch size",
                "type": "int",
                "min": 1,
                "placeholder": "128",
            },
            {
                "name": "torch_update_epochs",
                "label": "Update epochs",
                "type": "int",
                "min": 1,
                "placeholder": "6",
            },
            {
                "name": "torch_ppo_clip",
                "label": "PPO clip",
                "type": "float",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "placeholder": "0.2",
            },
            {
                "name": "torch_entropy_coef",
                "label": "Entropy coefficient",
                "type": "float",
                "min": 0,
                "max": 1,
                "step": 0.001,
                "placeholder": "0.01",
            },
            {
                "name": "torch_value_coef",
                "label": "Value coefficient",
                "type": "float",
                "min": 0,
                "max": 2,
                "step": 0.01,
                "placeholder": "0.5",
            },
            {
                "name": "torch_awr_beta",
                "label": "AWR beta",
                "type": "float",
                "min": 0,
                "max": 10,
                "step": 0.1,
                "placeholder": "1.0",
            },
            {
                "name": "torch_max_grad_norm",
                "label": "Max grad norm",
                "type": "float",
                "min": 0,
                "step": 0.1,
                "placeholder": "1.0",
            },
            {
                "name": "torch_hidden_dim",
                "label": "Hidden dimension",
                "type": "int",
                "min": 32,
                "placeholder": "768",
            },
            {
                "name": "torch_compile",
                "label": "torch.compile",
                "type": "checkbox",
            },
            {
                "name": "torch_amp",
                "label": "Automatic mixed precision",
                "type": "checkbox",
            },
            {
                "name": "torch_deterministic",
                "label": "Deterministic CUDA",
                "type": "checkbox",
            },
            {
                "name": "replay_buffer_capacity",
                "label": "Replay buffer capacity",
                "type": "int",
                "min": 1,
                "placeholder": "50000",
            },
            {
                "name": "replay_buffer_on_gpu",
                "label": "Replay buffer on GPU",
                "type": "checkbox",
            },
        ],
    },
    {
        "id": "moe",
        "title": "Mixture-of-experts serving",
        "description": "Expert bank size, residency, and swap penalties.",
        "workflows": ["moe"],
        "fields": [
            {
                "name": "moe_enabled",
                "label": "Enable MoE",
                "type": "checkbox",
            },
            {
                "name": "moe_num_experts",
                "label": "Number of experts",
                "type": "int",
                "min": 1,
                "placeholder": "16",
            },
            {
                "name": "moe_top_k",
                "label": "Top-k experts",
                "type": "int",
                "min": 1,
                "placeholder": "2",
            },
            {
                "name": "moe_gpu_resident_experts",
                "label": "GPU-resident experts",
                "type": "int",
                "min": 1,
                "placeholder": "8",
            },
            {
                "name": "moe_swap_penalty",
                "label": "Swap penalty",
                "type": "float",
                "min": 0,
                "step": 0.001,
                "placeholder": "0.015",
            },
            {
                "name": "moe_cache_miss_penalty",
                "label": "Cache miss penalty",
                "type": "float",
                "min": 0,
                "step": 0.01,
                "placeholder": "0.120",
            },
        ],
    },
    {
        "id": "online",
        "title": "Online serving RL",
        "description": "Exploration, canary traffic, replay, and drift guardrails.",
        "workflows": ["online"],
        "fields": [
            {
                "name": "online_learning",
                "label": "Enable online learning",
                "type": "checkbox",
            },
            {
                "name": "online_exploration_rate",
                "label": "Exploration rate",
                "type": "float",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "placeholder": "0.12",
            },
            {
                "name": "online_canary_ratio",
                "label": "Canary ratio",
                "type": "float",
                "min": 0,
                "max": 1,
                "step": 0.05,
                "placeholder": "0.50",
            },
            {
                "name": "online_update_interval",
                "label": "Update interval (requests)",
                "type": "int",
                "min": 1,
                "placeholder": "32",
            },
            {
                "name": "online_batch_size",
                "label": "Update batch size",
                "type": "int",
                "min": 1,
                "placeholder": "128",
            },
            {
                "name": "online_reward_guard",
                "label": "Reward guard threshold",
                "type": "float",
                "min": 0,
                "max": 2,
                "step": 0.05,
                "placeholder": "0.75",
            },
            {
                "name": "online_replay_capacity",
                "label": "Replay capacity",
                "type": "int",
                "min": 1,
                "placeholder": "2048",
            },
        ],
    },
    {
        "id": "continuous",
        "title": "Continuous task stream",
        "description": "Long-horizon streaming RL over a task sequence.",
        "workflows": ["continuous"],
        "fields": [
            {
                "name": "continuous_max_tasks",
                "label": "Max tasks",
                "type": "int",
                "min": 1,
                "placeholder": "50000",
                "cli_flag": "--max-tasks",
            },
            {
                "name": "continuous_learning_enabled",
                "label": "Enable continuous learning",
                "type": "checkbox",
            },
            {
                "name": "continuous_update_every_n_tasks",
                "label": "Update every N tasks",
                "type": "int",
                "min": 1,
                "placeholder": "1",
            },
            {
                "name": "continuous_eval_every_n_tasks",
                "label": "Eval every N tasks",
                "type": "int",
                "min": 1,
                "placeholder": "500",
            },
            {
                "name": "continuous_exploration_rate",
                "label": "Exploration rate",
                "type": "float",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "placeholder": "0.15",
            },
            {
                "name": "continuous_drift_reward_delta",
                "label": "Drift reward delta",
                "type": "float",
                "min": 0,
                "step": 0.1,
                "placeholder": "3.0",
            },
            {
                "name": "continuous_task_stream_mode",
                "label": "Task stream mode",
                "type": "choice",
                "choices": [
                    {
                        "value": "library_cycle",
                        "label": "library_cycle — random prompts",
                    },
                    {
                        "value": "sequential",
                        "label": "sequential — fixed curriculum order",
                    },
                    {
                        "value": "reward_adaptive",
                        "label": "reward_adaptive — follow reward paths",
                    },
                    {"value": "jsonl", "label": "jsonl — custom task file"},
                ],
                "default": "library_cycle",
            },
            {
                "name": "continuous_task_jsonl_path",
                "label": "Task JSONL path",
                "type": "text",
                "placeholder": "outputs/chat_tasks.jsonl",
                "help": "Required when stream mode is jsonl (chat UI writes here).",
            },
            {
                "name": "continuous_replay_capacity",
                "label": "Replay capacity",
                "type": "int",
                "min": 8,
                "placeholder": "4096",
            },
            {
                "name": "continuous_batch_size",
                "label": "Replay batch size",
                "type": "int",
                "min": 1,
                "placeholder": "64",
            },
            {
                "name": "continuous_min_replay_before_update",
                "label": "Min replay before update",
                "type": "int",
                "min": 1,
                "placeholder": "8",
            },
        ],
    },
    {
        "id": "privileged",
        "title": "Backend & privileged paths",
        "description": "Requires “Allow privileged --set overrides” in environment settings.",
        "workflows": sorted(_RUN_WORKFLOWS),
        "privileged": True,
        "fields": [
            {
                "name": "backend",
                "label": "Measurement backend",
                "type": "choice",
                "choices": [
                    {"value": "simulator", "label": "simulator"},
                    {"value": "llama_cpp", "label": "llama_cpp"},
                ],
                "default": "simulator",
                "privileged": True,
            },
            {
                "name": "training_backend",
                "label": "Training backend",
                "type": "choice",
                "choices": [
                    {"value": "python", "label": "python (stdlib)"},
                    {"value": "pytorch", "label": "pytorch"},
                ],
                "default": "python",
                "privileged": True,
            },
            {
                "name": "llama_cpp_binary",
                "label": "llama.cpp binary",
                "type": "text",
                "placeholder": "/path/to/llama-cli",
                "help": "Required when backend=llama_cpp (local GGUF measurements).",
                "privileged": True,
            },
            {
                "name": "llama_cpp_model",
                "label": "llama.cpp model (GGUF)",
                "type": "text",
                "placeholder": "/path/to/model.gguf",
                "help": "Default GGUF path; chat UI can override via Hugging Face route selection.",
                "privileged": True,
            },
            {
                "name": "resume_from_checkpoint",
                "label": "Resume checkpoint path",
                "type": "text",
                "placeholder": "outputs/checkpoints/…",
                "privileged": True,
            },
        ],
    },
]

_CLI_HANDLED_KEYS = frozenset(
    {
        "config",
        "training_episodes",
        "evaluation_episodes",
        "benchmark_training_episodes",
        "benchmark_evaluation_episodes",
        "run_name",
        "seed",
        "config_overrides",
        "preset",
        "seeds",
        "episodes",
        "sweep_config",
        "vary",
        "requests",
        "nvidia_ack",
        "nvidia_smi_path",
        "privileged_overrides",
        "cuda",
        "force_reinstall",
        "full",
        "no_torch",
        "no_nvidia",
        "accept_gpu_install",
    }
)

_FIELD_BY_NAME: dict[str, dict[str, Any]] = {}
for _group in RL_FIELD_GROUPS:
    for _field in _group["fields"]:
        _FIELD_BY_NAME[_field["name"]] = _field


def field_groups_for_workflow(workflow_id: str) -> list[dict[str, Any]]:
    """Return RL field groups visible for a workflow."""
    groups: list[dict[str, Any]] = []
    for group in RL_FIELD_GROUPS:
        workflows = group.get("workflows") or []
        if workflow_id not in workflows:
            continue
        groups.append(
            {
                "id": group["id"],
                "title": group["title"],
                "description": group.get("description", ""),
                "privileged": bool(group.get("privileged")),
                "fields": list(group["fields"]),
            }
        )
    return groups


def collect_rl_set_overrides(options: dict[str, Any]) -> list[str]:
    """Turn structured RL UI options into --set KEY=VALUE strings."""
    entries: list[str] = []
    for key, raw in options.items():
        if key in _CLI_HANDLED_KEYS:
            continue
        field = _FIELD_BY_NAME.get(key)
        if field is None or field.get("cli_flag"):
            continue
        if raw is None:
            continue
        if isinstance(raw, bool):
            value = "true" if raw else "false"
        elif isinstance(raw, (int, float)):
            value = str(raw)
        else:
            value = str(raw).strip()
            if not value:
                continue
        set_key = field.get("set_key", key)
        entries.append(f"{set_key}={value}")
    return entries


def apply_rl_cli_flags(command: list[str], options: dict[str, Any]) -> None:
    """Append workflow-specific CLI flags declared on RL fields."""
    for key, raw in options.items():
        field = _FIELD_BY_NAME.get(key)
        if field is None:
            continue
        cli_flag = field.get("cli_flag")
        if not cli_flag or raw is None or str(raw).strip() == "":
            continue
        command.extend([cli_flag, str(raw).strip()])
