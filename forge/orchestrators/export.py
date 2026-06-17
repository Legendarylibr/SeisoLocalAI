"""Export job orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path
from seiso.export.formats import ExportFormat, ExportOptions, export_checkpoint
from seiso.security import SecurityError


class ExportOrchestrator(Orchestrator):
    kind = "export"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = payload.get("user_id")
        if not user_id:
            raise PermissionError("user_id required for export")
        try:
            checkpoint = assert_user_path(self.sandbox_root, user_id, payload["checkpoint"])
        except SecurityError as exc:
            raise PermissionError(str(exc)) from exc

        formats = [ExportFormat(f) for f in payload.get("formats", ["merged"])]
        options = ExportOptions(
            checkpoint=checkpoint,
            output_dir=Path(payload.get("output_dir", self.sandbox_root / "exports" / user_id / job_id)),
            formats=formats,
            gguf_quantizations=payload.get("gguf_quantizations", ["q4_k_m"]),
            hub_repo=payload.get("hub_repo"),
            hub_token=payload.get("hub_token"),
            sandbox_root=self.sandbox_root,
        )

        self._emit_log(job_id, f"Exporting checkpoint: {checkpoint.name}")

        loop = asyncio.get_running_loop()

        def on_log(msg: str) -> None:
            self._emit_log(job_id, msg)

        results = await loop.run_in_executor(
            None,
            lambda: export_checkpoint(options, on_log=on_log),
        )
        paths = {k: str(v) for k, v in results.items()}
        self._emit_log(job_id, "Export complete")
        return {"outputs": paths}
