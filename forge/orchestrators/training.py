"""Training job orchestrator with multi-GPU subprocess support."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
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

_HF_TOKEN_LOCK = threading.Lock()


class TrainingOrchestrator(Orchestrator):
    kind = "training"
    resource_key = "gpu"

    def __init__(self, sandbox_root: Path) -> None:
        super().__init__(sandbox_root)
        self._metrics_persist_tasks: dict[str, asyncio.Task[None]] = {}
        self._on_metrics_persist: dict[str, Any] = {}

    def set_metrics_persister(self, job_id: str, callback) -> None:
        """Register async callback(job_id, metrics_snapshot) for DB persistence."""
        self._on_metrics_persist[job_id] = callback

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        import os

        hf_token = payload.get("hf_token")
        with _HF_TOKEN_LOCK:
            prev_hf_token = os.environ.get("HF_TOKEN")
            prev_hub_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            token_applied = False
            if hf_token:
                os.environ["HF_TOKEN"] = str(hf_token)
                os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
                token_applied = True
            try:
                return await self._execute_training(job_id, payload)
            finally:
                if token_applied:
                    if prev_hf_token is None:
                        os.environ.pop("HF_TOKEN", None)
                    else:
                        os.environ["HF_TOKEN"] = prev_hf_token
                    if prev_hub_token is None:
                        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
                    else:
                        os.environ["HUGGING_FACE_HUB_TOKEN"] = prev_hub_token

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

        prepare_for_gpu_task(
            task="training",
            job_id=job_id,
            log=lambda msg: self._emit_log(job_id, msg),
        )
        config = TrainConfig.model_validate(payload["config"])
        config.output_dir = Path(
            payload.get("output_dir", self.sandbox_root / "checkpoints" / job_id)
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
                    loop.call_soon_threadsafe(self._emit_log, job_id, line)

                training_future = loop.run_in_executor(
                    None,
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
                    with contextlib.suppress(asyncio.TimeoutError, Exception):
                        await asyncio.wait_for(
                            asyncio.shield(training_future), timeout=600
                        )
                    raise
                from seiso.training.cancel import is_requested

                if is_requested(job_id):
                    raise asyncio.CancelledError()

            metrics_path = config.output_dir / "metrics.jsonl"
            if metrics_path.exists():
                metrics_summary = self._load_metrics_summary(metrics_path)
            else:
                metrics_summary = self._summarize_buffered_metrics(job_id)
        finally:
            clear(job_id)
            release_after_task(
                reason="training complete",
                log=lambda msg: self._emit_log(job_id, msg),
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

        self._metrics_persist_tasks[job_id] = loop.create_task(_persist())

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
        cfg_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8"
        )
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
        )
        self.register_subprocess(job_id, proc)

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
                self._emit_log(job_id, text)

        code = await proc.wait()
        cfg_path.unlink(missing_ok=True)
        if code != 0:
            raise RuntimeError(f"Distributed training exited with code {code}")

        checkpoints = sorted(
            config.output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime
        )
        if checkpoints:
            return checkpoints[-1]
        return config.output_dir

    @staticmethod
    def _load_metrics_summary(metrics_path: Path) -> dict[str, Any]:
        losses: list[float] = []
        eval_losses: list[float] = []
        steps = 0
        try:
            with metrics_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    point = json.loads(line)
                    steps = max(steps, int(point.get("step", 0)))
                    if point.get("loss") is not None:
                        losses.append(float(point["loss"]))
                    if point.get("eval_loss") is not None:
                        eval_losses.append(float(point["eval_loss"]))
        except (OSError, json.JSONDecodeError, ValueError):
            return {}
        return {
            "total_steps": steps,
            "final_loss": losses[-1] if losses else None,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "final_eval_loss": eval_losses[-1] if eval_losses else None,
            "points": len(losses) + len(eval_losses),
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
        return {
            "total_steps": max((int(m.get("step", 0)) for m in training), default=0),
            "final_loss": losses[-1] if losses else None,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "final_eval_loss": eval_losses[-1] if eval_losses else None,
            "points": len(training),
            "updated_at": time.time(),
        }
