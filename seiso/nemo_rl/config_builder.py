"""Build NeMo RL Hydra overrides and a Seiso launch sidecar YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from seiso.nemo_rl.config import NeMoRLConfig, NeMoRLRecipe


def build_hydra_overrides(config: NeMoRLConfig) -> list[str]:
    """Project Seiso knobs into NeMo RL Hydra override strings."""
    overrides: list[str] = [
        f"policy.model_name={_hydra_quote(config.model_id)}",
        f"checkpointing.checkpoint_dir={_hydra_quote(str(config.output_dir.resolve()))}",
        f"cluster.gpus_per_node={int(config.gpus_per_node)}",
        f"cluster.num_nodes={int(config.num_nodes)}",
    ]

    recipe = (
        config.recipe
        if isinstance(config.recipe, NeMoRLRecipe)
        else NeMoRLRecipe(str(config.recipe))
    )
    grpo_rpp: int | None = None
    if recipe in {NeMoRLRecipe.GRPO, NeMoRLRecipe.SMOKE}:
        if config.max_steps is not None:
            overrides.append(f"grpo.max_num_steps={int(config.max_steps)}")
        # Always override generations/prompt so upstream recipe defaults of 1
        # cannot silently disable grouped GRPO advantages.
        grpo_rpp = int(config.rollouts_per_prompt) if config.rollouts_per_prompt is not None else 4
        if grpo_rpp < 2:
            raise ValueError(
                f"rollouts_per_prompt must be >= 2 for NeMo RL GRPO/smoke (got {grpo_rpp})"
            )
        overrides.append(f"grpo.num_generations_per_prompt={grpo_rpp}")
        if config.num_prompts_per_step is not None:
            overrides.append(f"grpo.num_prompts_per_step={int(config.num_prompts_per_step)}")
        overrides.append(f"grpo.seed={int(config.seed)}")

    if config.learning_rate is not None:
        # NeMo RL DTensor path: policy.optimizer.kwargs.lr
        overrides.append(f"policy.optimizer.kwargs.lr={float(config.learning_rate)}")

    if config.use_lora:
        # NeMo RL LoRA GRPO/DPO: enable when the base recipe supports it.
        overrides.append("policy.lora_cfg.enabled=true")

    for extra in config.extra_overrides:
        item = str(extra).strip()
        if not item:
            continue
        # Refuse extras that would undo the G>=2 GRPO invariant (Hydra last-wins).
        key, _, value = item.partition("=")
        if key.strip() == "grpo.num_generations_per_prompt":
            try:
                generations = int(value.strip())
            except ValueError as exc:
                raise ValueError(
                    f"invalid grpo.num_generations_per_prompt override: {item!r}"
                ) from exc
            if generations < 2:
                raise ValueError(
                    "extra_overrides cannot set grpo.num_generations_per_prompt < 2 "
                    f"(got {value!r}); grouped GRPO requires G>=2"
                )
        overrides.append(item)

    # Re-assert GRPO group size last so Hydra last-wins cannot undo it.
    if grpo_rpp is not None:
        overrides.append(f"grpo.num_generations_per_prompt={grpo_rpp}")
    return overrides


def write_launch_sidecar(config: NeMoRLConfig, *, nemo_root: Path) -> Path:
    """Write a Seiso-side launch record under ``output_dir`` for reproducibility."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "framework": "nemo_rl",
        "upstream": "https://github.com/NVIDIA-NeMo/RL",
        "model_id": config.model_id,
        "recipe": (
            config.recipe.value if isinstance(config.recipe, NeMoRLRecipe) else str(config.recipe)
        ),
        "nemo_rl_root": str(nemo_root),
        "base_config": config.recipe_base_config(),
        "script": config.recipe_script(),
        "gpus_per_node": config.gpus_per_node,
        "num_nodes": config.num_nodes,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "num_prompts_per_step": config.num_prompts_per_step,
        "seed": config.seed,
        "use_lora": config.use_lora,
        "hydra_overrides": build_hydra_overrides(config),
        "dry_run": config.dry_run,
    }
    path = out / "nemo_rl_launch.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def build_command(
    config: NeMoRLConfig,
    *,
    nemo_root: Path,
    uv: str,
) -> list[str]:
    """Assemble ``uv run python <script> --config <base> <overrides…>``."""
    script = config.recipe_script()
    base = config.recipe_base_config()
    script_path = nemo_root / script
    base_path = _resolve_base_config_path(nemo_root, base)
    if not script_path.is_file():
        raise FileNotFoundError(f"NeMo RL script missing: {script_path}")
    if not base_path.is_file():
        raise FileNotFoundError(f"NeMo RL base config missing: {base_path}")

    # Prefer a path relative to nemo_root for Hydra when possible.
    try:
        base_arg = str(base_path.resolve().relative_to(nemo_root.resolve()))
    except ValueError:
        base_arg = str(base_path)

    cmd = [
        uv,
        "run",
        "python",
        script,
        "--config",
        base_arg,
    ]
    cmd.extend(build_hydra_overrides(config))
    return cmd


def _resolve_base_config_path(nemo_root: Path, base: str) -> Path:
    """Resolve recipe base YAML under ``nemo_root``; reject ``..`` escapes."""
    from seiso.security import SecurityError, assert_relative_artifact_name, assert_within

    raw = str(base or "").strip()
    if not raw:
        raise ValueError("NeMo RL base_config must not be empty")
    root = nemo_root.resolve()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        try:
            return assert_within(root, candidate)
        except SecurityError as exc:
            raise ValueError(
                f"nemo_rl base_config must resolve under the NeMo checkout (rejected: {raw!r})"
            ) from exc
    try:
        assert_relative_artifact_name(raw, field="base_config")
    except ValueError as exc:
        raise ValueError(
            f"nemo_rl base_config must be a relative path without '..' (rejected: {raw!r})"
        ) from exc
    try:
        return assert_within(root, root / raw)
    except SecurityError as exc:
        raise ValueError(
            f"nemo_rl base_config must resolve under the NeMo checkout (rejected: {raw!r})"
        ) from exc


def _hydra_quote(value: str) -> str:
    """Quote Hydra override values that contain special characters."""
    text = str(value)
    if not text:
        return '""'
    needs = any(ch in text for ch in " =\\\"'{}[](),")
    if needs:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text
