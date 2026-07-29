"""Training job orchestrator with multi-GPU subprocess support."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.hardware import live_metrics
from seiso.models.hf_env import configure_hf_hub_cache
from seiso.training.config import TrainConfig, run_training
from seiso.training.metrics import parse_metric_line
from seiso.training.multi_gpu import (
    DistributedPlan,
    detect_training_layout,
    gpu_stats,
    launch_worker_command,
    resolve_distributed_plan,
)

# asyncio.Lock: serialize HF token env mutation without blocking the event loop
# (threading.Lock held across await deadlocks all Forge requests).
_HF_TOKEN_LOCK = asyncio.Lock()


class TrainingOrchestrator(Orchestrator):
    kind = "training"
    # GPU exclusivity: prepare_for_gpu_task + file lock + GPU_EXECUTOR (S1-009).
    resource_key = None

    def __init__(self, sandbox_root: Path) -> None:
        super().__init__(sandbox_root)
        self._metrics_persist_tasks: dict[str, asyncio.Task[None]] = {}
        self._on_metrics_persist: dict[str, Any] = {}

    def set_metrics_persister(self, job_id: str, callback) -> None:
        """Register async callback(job_id, metrics_snapshot) for DB persistence."""
        self._on_metrics_persist[job_id] = callback

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Serialize training jobs that may apply a per-user HF token. Do not
        # mutate process-wide HF_TOKEN for the whole await — the trainer applies
        # the token at model-load time via config.extra (workers get it in env).
        async with _HF_TOKEN_LOCK:
            return await self._execute_training(job_id, payload)

    async def cancel(self, job_id: str) -> bool:
        from seiso.training.cancel import request

        request(job_id)
        return await super().cancel(job_id)

    async def _execute_training(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        configure_hf_hub_cache(self.sandbox_root)
        from forge.services.memory_release import (
            prepare_for_gpu_task,
            release_after_task,
        )

        config = TrainConfig.model_validate(payload["config"])
        if payload.get("hf_token"):
            config.extra = {**(config.extra or {}), "hf_token": str(payload["hf_token"])}
        user_id = str(payload.get("user_id") or "")
        if "output_dir" in payload and payload["output_dir"]:
            from seiso.security import assert_user_scoped_path, assert_within

            out = Path(payload["output_dir"])
            if user_id:
                # Re-check even when the HTTP route set the path (defense in depth).
                config.output_dir = assert_user_scoped_path(
                    self.sandbox_root, user_id, out
                )
            else:
                config.output_dir = assert_within(self.sandbox_root, out)
        elif user_id:
            from seiso.security import safe_join

            config.output_dir = safe_join(
                self.sandbox_root, "checkpoints", user_id, job_id
            )
        else:
            raise ValueError(
                "output_dir is required when user_id is missing "
                "(refusing unscoped checkpoints/)"
            )

        multi_gpu = bool(payload.get("multi_gpu", config.multi_gpu))
        config.multi_gpu = multi_gpu
        config.use_triton = bool(
            payload.get("use_triton", config.extra.get("use_triton", config.use_triton))
        )
        config.use_fused_ce = bool(
            payload.get(
                "use_fused_ce", config.extra.get("use_fused_ce", config.use_fused_ce)
            )
        )
        config.use_fused_lora = bool(
            payload.get(
                "use_fused_lora",
                config.extra.get("use_fused_lora", config.use_fused_lora),
            )
        )

        layout = detect_training_layout()
        distributed_plan = resolve_distributed_plan(config, layout)
        self._emit_log(job_id, f"Starting training: {config.model_id}")
        if resolved := config.extra.get("resolved_model_path"):
            self._emit_log(job_id, f"Using cached weights: {resolved}")
        self._emit_log(
            job_id, f"Method: {config.method.value}, quant: {config.quant.value}"
        )
        self._emit_log(
            job_id,
            f"GPUs: {layout.device_count} visible, world_size={layout.world_size}, "
            f"multi_gpu={multi_gpu}, distributed={distributed_plan.strategy}, "
            f"fused_kernels={config.use_triton}, fused_ce={config.use_fused_ce}",
        )

        prepare_for_gpu_task(
            task="training",
            job_id=job_id,
            log=lambda msg: self._emit_log(job_id, msg),
        )
        loop = asyncio.get_running_loop()
        stop_poll = asyncio.Event()
        poll_task = asyncio.create_task(self._poll_system_metrics(job_id, stop_poll))
        metrics_summary: dict[str, Any] = {}
        from seiso.training.cancel import clear, register

        register(job_id)
        try:
            if distributed_plan.enabled:
                checkpoint = await self._run_distributed(
                    job_id,
                    config,
                    distributed_plan,
                    hf_token=payload.get("hf_token"),
                )
            else:
                if multi_gpu:
                    self._emit_log(
                        job_id,
                        f"distributed launch skipped: {distributed_plan.reason}",
                    )

                def on_metric(metric: dict[str, Any]) -> None:
                    loop.call_soon_threadsafe(self._emit_metric, job_id, metric)
                    self._schedule_metrics_persist(job_id, loop)

                def on_log(line: str) -> None:
                    if line.startswith("MEMORY_POLICY "):
                        try:
                            policy = json.loads(line.removeprefix("MEMORY_POLICY "))
                            loop.call_soon_threadsafe(
                                self._emit_event, job_id, "memory_policy", policy
                            )
                        except json.JSONDecodeError:
                            pass
                    loop.call_soon_threadsafe(self._emit_log, job_id, line)

                from forge.services.executors import GPU_EXECUTOR

                training_future = loop.run_in_executor(
                    GPU_EXECUTOR,
                    lambda: run_training(
                        config,
                        on_metric=on_metric,
                        on_log=on_log,
                        job_id=job_id,
                    ),
                )
                try:
                    checkpoint = await training_future
                except asyncio.CancelledError:
                    from seiso.training.cancel import request

                    request(job_id)
                    # Await completion before GPU release in finally (no timeout).
                    with contextlib.suppress(Exception):
                        await asyncio.shield(training_future)
                    raise
                except Exception as exc:
                    # Cooperative slime/NeMo cancel raises InterruptedError in-thread.
                    if isinstance(exc, InterruptedError) or (
                        isinstance(exc, RuntimeError)
                        and "cancelled" in str(exc).lower()
                    ):
                        raise asyncio.CancelledError() from exc
                    # Executor wraps some exceptions; unwrap common cancel signals.
                    cause = getattr(exc, "__cause__", None) or getattr(
                        exc, "__context__", None
                    )
                    if isinstance(cause, InterruptedError):
                        raise asyncio.CancelledError() from exc
                    raise
                from seiso.training.cancel import is_requested

                if is_requested(job_id):
                    raise asyncio.CancelledError()

            metrics_path = config.output_dir / "metrics.jsonl"
            if not metrics_path.exists():
                metrics_path = config.output_dir / "slime_single_gpu_metrics.jsonl"
            if metrics_path.exists():
                metrics_summary = self._load_metrics_summary(metrics_path)
            else:
                metrics_summary = self._summarize_buffered_metrics(job_id)
        finally:
            clear(job_id)
            release_after_task(
                reason="training complete",
                log=lambda msg: self._emit_log(job_id, msg),
                job_id=job_id,
            )
            stop_poll.set()
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
            persist = self._on_metrics_persist.pop(job_id, None)
            if persist:
                await persist(job_id, self.get_metrics(job_id), metrics_summary)

        stats = gpu_stats()
        if stats:
            self._emit_log(job_id, f"GPU utilization snapshot: {len(stats)} device(s)")

        self._emit_log(job_id, f"Checkpoint saved: {checkpoint}")
        return {
            "checkpoint_path": str(checkpoint),
            "metrics_summary": metrics_summary,
        }

    def _schedule_metrics_persist(
        self, job_id: str, loop: asyncio.AbstractEventLoop
    ) -> None:
        task = self._metrics_persist_tasks.get(job_id)
        if task and not task.done():
            return

        async def _persist() -> None:
            await asyncio.sleep(3)
            callback = self._on_metrics_persist.get(job_id)
            if callback:
                await callback(job_id, self.get_metrics(job_id), {})

        task = loop.create_task(_persist())

        def _log_persist_failure(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                return
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    "Failed to persist training metrics for job %s", job_id
                )

        task.add_done_callback(_log_persist_failure)
        self._metrics_persist_tasks[job_id] = task

    async def _poll_system_metrics(self, job_id: str, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                snapshot = live_metrics()
                snapshot["type"] = "system"
                self._emit_metric(job_id, snapshot)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=2.0)
                break
            except asyncio.TimeoutError:
                continue

    async def _run_distributed(
        self,
        job_id: str,
        config: TrainConfig,
        plan: DistributedPlan,
        *,
        hf_token: str | None = None,
    ) -> Path:
        config.multi_gpu = True
        import yaml

        config.output_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = config.output_dir / f"{job_id}.worker.yaml"
        # Never persist HF tokens in the worker YAML (env injection below).
        dump_cfg = config.model_dump(mode="json")
        extra = dump_cfg.get("extra")
        if isinstance(extra, dict) and "hf_token" in extra:
            extra = {k: v for k, v in extra.items() if k != "hf_token"}
            dump_cfg["extra"] = extra
        cfg_path.write_text(yaml.safe_dump(dump_cfg), encoding="utf-8")
        cfg_path.chmod(0o600)

        self._emit_log(
            job_id,
            f"Launching accelerate --multi_gpu --num_processes={plan.world_size} "
            f"--num_machines={plan.nnodes} --machine_rank={plan.node_rank}",
        )

        cmd = launch_worker_command(str(cfg_path), plan)
        env = {**__import__("os").environ, "SEISO_EMIT_METRICS_STDOUT": "1"}
        if hf_token:
            env["HF_TOKEN"] = str(hf_token)
            env.pop("HUGGING_FACE_HUB_TOKEN", None)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self.sandbox_root),
            env=env,
            start_new_session=os.name == "posix",
        )
        self.register_subprocess(job_id, proc, process_group=os.name == "posix")

        assert proc.stdout
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode().rstrip()
            metric = parse_metric_line(text)
            if metric:
                self._emit_metric(job_id, metric)
                loop = asyncio.get_running_loop()
                self._schedule_metrics_persist(job_id, loop)
            else:
                if text.startswith("MEMORY_POLICY "):
                    with contextlib.suppress(json.JSONDecodeError):
                        self._emit_event(
                            job_id,
                            "memory_policy",
                            json.loads(text.removeprefix("MEMORY_POLICY ")),
                        )
                self._emit_log(job_id, text)

        code = await proc.wait()
        cfg_path.unlink(missing_ok=True)
        if code != 0:
            from seiso.training.cancel import is_requested

            rec = self.get_job(job_id)
            if is_requested(job_id) or (rec is not None and rec.cancel_requested):
                raise asyncio.CancelledError()
            raise RuntimeError(f"Distributed training exited with code {code}")

        artifact = self._resolve_distributed_artifact(config.output_dir)
        if artifact is not None:
            return artifact
        raise RuntimeError(
            "Distributed training exited successfully but wrote no "
            f"usable checkpoint under {config.output_dir}"
        )

    @staticmethod
    def _looks_like_saved_model(path: Path) -> bool:
        if not path.is_dir():
            return False
        markers = (
            "adapter_config.json",
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        if any((path / name).is_file() for name in markers):
            return True
        return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))

    @classmethod
    def _resolve_distributed_artifact(cls, output_dir: Path) -> Path | None:
        """Prefer slime/NeMo final paths; fall back to newest checkpoint-* / root."""
        root = Path(output_dir)
        if not root.is_dir():
            return None

        state_path = root / "slime_training_state.json"
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                state = {}
            for key in ("final_checkpoint_dir", "best_checkpoint_dir"):
                raw = state.get(key)
                if not raw:
                    continue
                candidate = Path(str(raw))
                if not candidate.is_absolute():
                    candidate = root / candidate
                try:
                    candidate = candidate.resolve()
                    root.resolve()
                    candidate.relative_to(root.resolve())
                except ValueError:
                    continue
                if cls._looks_like_saved_model(candidate):
                    return candidate

        # Slime writes the final weights at output_dir; SFT uses checkpoint-*.
        # Prefer the newest path that looks like a saved model.
        candidates: list[Path] = []
        if cls._looks_like_saved_model(root):
            candidates.append(root)
        for path in root.glob("checkpoint-*"):
            if path.is_dir() and cls._looks_like_saved_model(path):
                candidates.append(path)
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)

        checkpoints = [p for p in root.glob("checkpoint-*") if p.is_dir()]
        if checkpoints:
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
        return None

    @staticmethod
    def _load_metrics_summary(metrics_path: Path) -> dict[str, Any]:
        losses: list[float] = []
        eval_losses: list[float] = []
        rewards: list[float] = []
        steps = 0
        points = 0
        try:
            with metrics_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    points += 1
                    point = json.loads(line)
                    steps = max(steps, int(point.get("step", 0)))
                    if point.get("loss") is not None:
                        losses.append(float(point["loss"]))
                    if point.get("eval_loss") is not None:
                        eval_losses.append(float(point["eval_loss"]))
                    reward = point.get("reward")
                    if reward is None:
                        reward = point.get("reward_mean")
                    if reward is not None:
                        rewards.append(float(reward))
        except (OSError, json.JSONDecodeError, ValueError):
            return {}
        return {
            "total_steps": steps,
            "final_loss": losses[-1] if losses else None,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "final_eval_loss": eval_losses[-1] if eval_losses else None,
            "final_reward": rewards[-1] if rewards else None,
            "best_reward": max(rewards) if rewards else None,
            "points": points,
            "updated_at": time.time(),
        }

    def _summarize_buffered_metrics(self, job_id: str) -> dict[str, Any]:
        training = [
            m
            for m in self.get_metrics(job_id)
            if m.get("type") in ("training", "eval") and m.get("type") != "system"
        ]
        losses = [float(m["loss"]) for m in training if m.get("loss") is not None]
        eval_losses = [
            float(m["eval_loss"]) for m in training if m.get("eval_loss") is not None
        ]
        rewards: list[float] = []
        for m in training:
            if m.get("reward") is not None:
                rewards.append(float(m["reward"]))
            elif m.get("reward_mean") is not None:
                rewards.append(float(m["reward_mean"]))
        return {
            "total_steps": max((int(m.get("step", 0)) for m in training), default=0),
            "final_loss": losses[-1] if losses else None,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "final_eval_loss": eval_losses[-1] if eval_losses else None,
            "final_reward": rewards[-1] if rewards else None,
            "best_reward": max(rewards) if rewards else None,
            "points": len(training),
            "updated_at": time.time(),
        }
