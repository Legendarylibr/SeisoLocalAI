"""Launch NVIDIA NeMo RL as an external post-training process."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seiso.nemo_rl.bootstrap import resolve_nemo_rl_root, resolve_uv_executable
from seiso.nemo_rl.config import NeMoRLConfig, NeMoRLRecipe
from seiso.nemo_rl.config_builder import build_command, write_launch_sidecar

logger = logging.getLogger(__name__)


def train_nemo_rl(config: NeMoRLConfig) -> Path:
    """Run (or dry-run) a NeMo RL recipe; return the Seiso output directory."""
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.sandbox_root is not None:
        from seiso.security import assert_within

        assert_within(Path(config.sandbox_root), output_dir)

    try:
        nemo_root = resolve_nemo_rl_root(config.nemo_rl_root)
    except FileNotFoundError:
        if not config.dry_run:
            raise
        # Dry-run may preview Hydra overrides without a checkout present.
        nemo_root = Path(
            config.nemo_rl_root
            or os.environ.get("SEISO_NEMO_RL_ROOT")
            or "<nemo-rl-not-found>"
        )

    sidecar = write_launch_sidecar(config, nemo_root=nemo_root)

    if config.dry_run:
        logger.info("NeMo RL dry-run: wrote launch sidecar %s", sidecar)
        _write_manifest(config, output_dir, nemo_root=nemo_root, status="dry_run")
        return output_dir

    uv = resolve_uv_executable(config.uv_executable)
    cmd = build_command(config, nemo_root=nemo_root, uv=uv)
    env = os.environ.copy()
    # Keep HF / WANDB settings from the Seiso process when present.
    logger.info("Launching NeMo RL: cwd=%s cmd=%s", nemo_root, " ".join(cmd))

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(nemo_root),
            env=env,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to launch NeMo RL: {exc}") from exc

    if completed.returncode != 0:
        raise RuntimeError(
            f"NeMo RL exited with code {completed.returncode}. "
            f"See {sidecar} for the exact launch command and overrides."
        )

    _write_manifest(config, output_dir, nemo_root=nemo_root, status="completed")
    return output_dir


def _write_manifest(
    config: NeMoRLConfig,
    output_dir: Path,
    *,
    nemo_root: Path,
    status: str,
) -> None:
    recipe = (
        config.recipe.value
        if isinstance(config.recipe, NeMoRLRecipe)
        else str(config.recipe)
    )
    payload: dict[str, Any] = {
        "model_id": config.model_id,
        "method": "nemo_rl",
        "framework": "nemo_rl",
        "upstream": "https://github.com/NVIDIA-NeMo/RL",
        "post_training_algorithm": f"nemo_rl_{recipe}",
        "recipe": recipe,
        "nemo_rl_root": str(nemo_root),
        "base_config": config.recipe_base_config(),
        "gpus_per_node": config.gpus_per_node,
        "num_nodes": config.num_nodes,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "num_prompts_per_step": config.num_prompts_per_step,
        "seed": config.seed,
        "use_lora": config.use_lora,
        "adapter": "lora" if config.use_lora else "full",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "launch_sidecar": str(output_dir / "nemo_rl_launch.yaml"),
    }
    path = output_dir / "seiso_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
