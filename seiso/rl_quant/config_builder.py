"""Build FrameworkConfig for Seiso RL quantization jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seiso.rl_quant.bootstrap import _VENDOR_ROOT, ensure_adaptive_quant_importable


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
    """Return adaptive_quant.configuration.FrameworkConfig for a Forge job."""
    ensure_adaptive_quant_importable()
    from adaptive_quant.configuration import config_to_flat_dict
    from adaptive_quant.easy_config import config_from_dict, load_config, named_preset

    run_name = str(payload.get("run_name") or f"seiso_{job_id[:8]}")
    output_root = data_dir / "rl_quant" / user_id / job_id
    output_root.mkdir(parents=True, exist_ok=True)

    if config_file := payload.get("config_file"):
        path = Path(config_file)
        if not path.is_file():
            path = _VENDOR_ROOT / "configs" / config_file
        base = load_config(path)
    else:
        base = named_preset(str(payload.get("preset", "reproducible")))

    preset = str(payload.get("preset", "")).lower()
    write_report = payload.get("write_research_report")
    if write_report is None:
        write_report = preset not in {"minimal", "smoke"}

    overrides: dict[str, Any] = {
        **_artifact_paths(output_root, run_name),
        "training_episodes": int(payload.get("training_episodes", base.training_episodes)),
        "evaluation_episodes": int(payload.get("evaluation_episodes", base.evaluation_episodes)),
        "seed": int(payload.get("seed", base.seed)),
        "backend": str(payload.get("backend", base.backend)),
        "training_backend": str(payload.get("training_backend", base.training_backend)),
        "write_research_report": bool(write_report),
        "llama_cpp_gguf_export_enabled": bool(payload.get("gguf_export", False)),
    }

    if preset in {"post_train", "posttrain"}:
        overrides["prompt_library_path"] = str(_VENDOR_ROOT / "prompts" / "post_train_library.json")
    elif payload.get("prompt_library"):
        overrides["prompt_library_path"] = str(payload["prompt_library"])

    if reward := payload.get("reward_weights"):
        overrides["reward_weights"] = reward

    if checkpoint := payload.get("checkpoint_path"):
        overrides["external_quality_path"] = str(checkpoint)

    if gguf := payload.get("gguf_path"):
        overrides["llama_cpp_model"] = str(gguf)
        overrides["backend"] = "llama_cpp"
        if payload.get("gguf_export"):
            overrides["llama_cpp_gguf_export_source"] = str(gguf)

    if binary := payload.get("llama_cpp_binary"):
        overrides["llama_cpp_binary"] = str(binary)

    if payload.get("moe_enabled") is True:
        overrides["moe_enabled"] = True

    flat = config_to_flat_dict(base)
    flat.update(overrides)
    return config_from_dict(flat, base=base, strict=False)
