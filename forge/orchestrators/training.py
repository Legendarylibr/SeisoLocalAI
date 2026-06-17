"""Training job orchestrator with multi-GPU subprocess support."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from seiso.training.config import TrainConfig, run_training
from seiso.training.multi_gpu import detect_gpus, gpu_stats, launch_worker_command


class TrainingOrchestrator(Orchestrator):
    kind = "training"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        config = TrainConfig.model_validate(payload["config"])
        config.output_dir = Path(payload.get("output_dir", self.sandbox_root / "checkpoints" / job_id))

        multi_gpu = bool(payload.get("multi_gpu", config.multi_gpu))
        config.multi_gpu = multi_gpu
        config.use_triton = bool(
            payload.get("use_triton", config.extra.get("use_triton", config.use_triton))
        )

        layout = detect_gpus()
        self._emit_log(job_id, f"Starting training: {config.model_id}")
        self._emit_log(job_id, f"Method: {config.method.value}, quant: {config.quant.value}")
        self._emit_log(
            job_id,
            f"GPUs: {layout.device_count} visible, world_size={layout.world_size}, "
            f"multi_gpu={multi_gpu}, triton={config.use_triton}",
        )

        if multi_gpu and layout.device_count > 1:
            checkpoint = await self._run_distributed(job_id, config, layout.device_count)
        else:
            if multi_gpu and layout.device_count <= 1:
                self._emit_log(job_id, "multi_gpu requested but only one GPU — running single-process")
            loop = asyncio.get_running_loop()
            checkpoint = await loop.run_in_executor(None, run_training, config)

        stats = gpu_stats()
        if stats:
            self._emit_log(job_id, f"GPU stats: {json.dumps(stats)}")

        self._emit_log(job_id, f"Checkpoint saved: {checkpoint}")
        return {"checkpoint_path": str(checkpoint), "gpu_stats": stats}

    async def _run_distributed(self, job_id: str, config: TrainConfig, nproc: int) -> Path:
        config.multi_gpu = True
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(config.model_dump(mode="json"), f)
            cfg_path = f.name

        self._emit_log(job_id, f"Launching torchrun --nproc_per_node={nproc}")

        cmd = launch_worker_command(cfg_path, nproc)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self.sandbox_root),
        )
        self.register_subprocess(job_id, proc)

        assert proc.stdout
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            self._emit_log(job_id, line.decode().rstrip())

        code = await proc.wait()
        Path(cfg_path).unlink(missing_ok=True)
        if code != 0:
            raise RuntimeError(f"Distributed training exited with code {code}")

        checkpoints = sorted(config.output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            return checkpoints[-1]
        return config.output_dir
