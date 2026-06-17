"""RL quantization job orchestrator — adaptive_quant research pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path
from seiso.rl_quant.runner import run_rl_quant_job
from seiso.security import SecurityError


class RLQuantOrchestrator(Orchestrator):
    kind = "rl_quant"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = payload.get("user_id")
        if not user_id:
            raise PermissionError("user_id required for RL quant job")

        if checkpoint := payload.get("checkpoint_path"):
            try:
                assert_user_path(self.sandbox_root, user_id, checkpoint)
            except SecurityError as exc:
                raise PermissionError(str(exc)) from exc

        if gguf := payload.get("gguf_path"):
            try:
                assert_user_path(self.sandbox_root, user_id, gguf)
            except SecurityError as exc:
                raise PermissionError(str(exc)) from exc

        self._emit_log(job_id, "Starting adaptive RL quantization pipeline")

        loop = asyncio.get_running_loop()

        def on_log(msg: str) -> None:
            self._emit_log(job_id, msg)

        result = await loop.run_in_executor(
            None,
            lambda: run_rl_quant_job(
                job_id=job_id,
                user_id=user_id,
                data_dir=Path(self.sandbox_root),
                payload=payload,
                on_log=on_log,
            ),
        )
        self._emit_log(job_id, f"Artifacts: {result.get('output_dir')}")
        return result
