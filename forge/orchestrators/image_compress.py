"""Stable Diffusion image compression job orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path
from seiso.image_compress.runner import run_image_compress_job
from seiso.security import SecurityError


class ImageCompressOrchestrator(Orchestrator):
    kind = "image_compress"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = payload.get("user_id")
        if not user_id:
            raise PermissionError("user_id required for image compression job")

        if model_dir := payload.get("model_dir"):
            try:
                assert_user_path(self.sandbox_root, user_id, model_dir)
            except SecurityError as exc:
                raise PermissionError(str(exc)) from exc

        if data_path := payload.get("data_path"):
            try:
                assert_user_path(self.sandbox_root, user_id, data_path)
            except SecurityError as exc:
                raise PermissionError(str(exc)) from exc

        self._emit_log(job_id, "Starting Stable Diffusion image compression pipeline")

        loop = asyncio.get_running_loop()

        def on_log(msg: str) -> None:
            self._emit_log(job_id, msg)

        result = await loop.run_in_executor(
            None,
            lambda: run_image_compress_job(
                job_id=job_id,
                user_id=user_id,
                data_dir=Path(self.sandbox_root),
                payload=payload,
                on_log=on_log,
            ),
        )
        self._emit_log(job_id, f"Run directory: {result.get('run_dir')}")
        return result
