"""Build FrameworkConfig for Seiso RL quantization jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seiso.bundled.config_builder import job_output_root, resolve_config_file_path
from seiso.rl_quant.bootstrap import bundle_root, ensure_adaptive_quant_importable
from seiso.rl_quant.presets import lookup_preset


def _artifact_paths(output_root: Path, run_name: str) -> dict[str, str]:
    root = str(output_root.resolve())
    return {
        "outputs_dir": root,
        "log_dir": f"{root}/logs",
        "benchmark_dir": f"{root}/benchmarks",
        "analysis_dir": f"{root}/analysis",
        "checkpoint_dir": f"{root}/checkpoints",
        "report_dir": f"{root}/reports",
        "gguf_export_dir": f"{root}/gguf",
        "run_name": run_name,
    }


def build_framework_config(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
) -> Any:
    """Return seiso.adaptive_quant.configuration.FrameworkConfig for a Forge job."""
    ensure_adaptive_quant_importable()
    from seiso.adaptive_quant.configuration import config_to_flat_dict
    from seiso.adaptive_quant.easy_config import (
        config_from_dict,
        load_config,
        named_preset,
    )

    run_name = str(payload.get("run_name") or f"seiso_{job_id[:8]}")
    output_root = job_output_root(data_dir, "rl_quant", user_id, job_id)
    product = lookup_preset(payload.get("preset"))

    if config_file := payload.get("config_file"):
        path = resolve_config_file_path(config_file, bundle_root=bundle_root())
        if path is None:
            raise ValueError(f"Config file not found: {config_file}")
        base = load_config(path)
    else:
        named = (
            product.resolve_named_preset()
            if product is not None
            else str(payload.get("preset", "reproducible"))
        )
        base = named_preset(named)

    # Product registry owns Seiso defaults (simulator/python). Research named_preset
    # may still supply continuous/router knobs for post_train under those backends.
    default_backend = product.backend if product is not None else base.backend
    default_training_backend = (
        product.training_backend if product is not None else base.training_backend
    )

    overrides: dict[str, Any] = {
        **_artifact_paths(output_root, run_name),
        "training_episodes": int(
            payload.get("training_episodes", base.training_episodes)
        ),
        "evaluation_episodes": int(
            payload.get("evaluation_episodes", base.evaluation_episodes)
        ),
        "seed": int(payload.get("seed", base.seed)),
        "backend": str(payload.get("backend", default_backend)),
        "training_backend": str(
            payload.get("training_backend", default_training_backend)
        ),
        "llama_cpp_gguf_export_enabled": bool(payload.get("gguf_export", False)),
    }

    if product is not None and product.prompt_library:
        overrides["prompt_library_path"] = str(bundle_root() / product.prompt_library)
    elif payload.get("prompt_library"):
        overrides["prompt_library_path"] = str(payload["prompt_library"])

    if reward := payload.get("reward_weights"):
        overrides["reward_weights"] = reward

    if checkpoint := payload.get("checkpoint_path"):
        # Model/policy checkpoint — not a quality sidecar JSON.
        overrides["resume_from_checkpoint"] = str(checkpoint)
    if quality := payload.get("external_quality_path") or payload.get("quality_sidecar"):
        overrides["external_quality_path"] = str(quality)

    if gguf := payload.get("gguf_path"):
        overrides["llama_cpp_model"] = str(gguf)
        overrides["backend"] = "llama_cpp"
        if payload.get("gguf_export"):
            overrides["llama_cpp_gguf_export_source"] = str(gguf)

    if binary := payload.get("llama_cpp_binary"):
        overrides["llama_cpp_binary"] = str(binary)

    if payload.get("moe_enabled") is True:
        overrides["moe_enabled"] = True

    if payload.get("kernel_rl_enabled") is True:
        overrides["kernel_rl_enabled"] = True
    if (kernel_cfg := payload.get("kernel")) and isinstance(kernel_cfg, dict):
        overrides.update(
            {
                f"kernel_{key}": value
                for key, value in kernel_cfg.items()
                if key != "rl_enabled"
            }
        )
        if kernel_cfg.get("rl_enabled") is True:
            overrides["kernel_rl_enabled"] = True
    for flat_key in (
        "kernel_live_benchmark",
        "kernel_hidden_dim",
        "kernel_batch_rows",
        "kernel_benchmark_every_n_episodes",
        "kernel_default_profile",
        "kernel_profile_count",
    ):
        if flat_key in payload and payload[flat_key] is not None:
            overrides[flat_key] = payload[flat_key]

    if payload.get("router_enabled") is True:
        overrides["router_enabled"] = True
    if (
        (routes := payload.get("router_routes"))
        and isinstance(routes, (list, tuple))
        and routes
    ):
        overrides["router_routes"] = tuple(str(r) for r in routes)
    if (
        (modes := payload.get("hardware_modes"))
        and isinstance(modes, (list, tuple))
        and modes
    ):
        overrides["hardware_modes"] = tuple(str(m) for m in modes)
    if (
        (repos := payload.get("route_hf_allowed_repos"))
        and isinstance(repos, (list, tuple))
        and repos
    ):
        overrides["route_hf_allowed_repos"] = tuple(str(r) for r in repos)
    for bound_key in (
        "router_exploration",
        "router_regression_penalty",
        "llama_cpp_timeout_s",
    ):
        if bound_key in payload and payload[bound_key] is not None:
            overrides[bound_key] = payload[bound_key]

    flat = config_to_flat_dict(base)
    flat.update(overrides)
    from seiso.memory.protection import apply_rl_memory_guards

    flat = apply_rl_memory_guards(flat)
    return config_from_dict(flat, base=base, strict=False)
