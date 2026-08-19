"""Launch NVIDIA NeMo RL as an external post-training process."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seiso.nemo_rl.bootstrap import resolve_nemo_rl_root, resolve_uv_executable
from seiso.nemo_rl.config import NeMoRLConfig, NeMoRLRecipe
from seiso.nemo_rl.config_builder import build_command, write_launch_sidecar

logger = logging.getLogger(__name__)


def train_nemo_rl(
    config: NeMoRLConfig,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Run (or dry-run) a NeMo RL recipe; return the Seiso output directory."""
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_fn = should_stop or (lambda: False)

    if config.sandbox_root is not None:
        from seiso.security import assert_within

        sandbox = Path(config.sandbox_root)
        assert_within(sandbox, output_dir)
        # Path-like Hydra overrides must stay inside the Forge user sandbox (NEMO-01).
        for item in config.extra_overrides or ():
            text = str(item)
            if "=" not in text:
                continue
            _key, _, value = text.partition("=")
            value = value.strip().strip("'\"")
            if not value or value.startswith(("http://", "https://", "s3://")):
                continue
            candidate = Path(value).expanduser()
            looks_like_path = (
                candidate.is_absolute()
                or value.startswith(("~/", "./", "../"))
                or "/" in value
                or "\\" in value
                or ".." in candidate.parts
            )
            if looks_like_path:
                # Resolve relative values against cwd (NeMo launch dir) so
                # embedded ``..`` segments cannot skip the sandbox check.
                target = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
                assert_within(sandbox, target)

    try:
        nemo_root = resolve_nemo_rl_root(config.nemo_rl_root)
    except FileNotFoundError:
        if not config.dry_run:
            raise
        # Dry-run may preview Hydra overrides without a checkout present.
        nemo_root = Path(
            config.nemo_rl_root or os.environ.get("SEISO_NEMO_RL_ROOT") or "<nemo-rl-not-found>"
        )

    sidecar = write_launch_sidecar(config, nemo_root=nemo_root)

    if config.dry_run:
        logger.info("NeMo RL dry-run: wrote launch sidecar %s", sidecar)
        _write_manifest(config, output_dir, nemo_root=nemo_root, status="dry_run")
        return output_dir

    if stop_fn():
        raise InterruptedError("NeMo RL cancelled before launch")

    uv = resolve_uv_executable(config.uv_executable)
    cmd = build_command(config, nemo_root=nemo_root, uv=uv)
    env = os.environ.copy()
    # Keep HF / WANDB settings from the Seiso process when present.
    logger.info("Launching NeMo RL: cwd=%s cmd=%s", nemo_root, " ".join(cmd))

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(nemo_root),
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to launch NeMo RL: {exc}") from exc

    code: int | None = None
    try:
        while True:
            if stop_fn():
                _terminate_process_group(proc)
                raise InterruptedError("NeMo RL cancelled")
            code = proc.poll()
            if code is not None:
                break
            time.sleep(0.5)
    except InterruptedError:
        raise
    except Exception:
        _terminate_process_group(proc)
        raise

    if code != 0:
        raise RuntimeError(
            f"NeMo RL exited with code {code}. "
            f"See {sidecar} for the exact launch command and overrides."
        )

    _write_manifest(config, output_dir, nemo_root=nemo_root, status="completed")
    return output_dir


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    """Best-effort terminate of the NeMo RL process group on cancel."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(Exception):
            proc.terminate()
    deadline = time.monotonic() + 10.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(Exception):
                proc.kill()


def _write_manifest(
    config: NeMoRLConfig,
    output_dir: Path,
    *,
    nemo_root: Path,
    status: str,
) -> None:
    recipe = config.recipe.value if isinstance(config.recipe, NeMoRLRecipe) else str(config.recipe)
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
