"""Export job orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.ollama_export import build_ollama_create_commands
from forge.services.user_paths import assert_user_path
from seiso.export.model_card import HubModelMetadata
from seiso.export.pipeline import prepare_export, run_export_plan
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

        hub_meta_raw = payload.get("hub_metadata")
        hub_metadata = HubModelMetadata(**hub_meta_raw) if hub_meta_raw else None
        output_dir = Path(payload.get("output_dir", self.sandbox_root / "exports" / user_id / job_id))

        self._emit_log(job_id, f"Exporting checkpoint: {checkpoint.name}")

        loop = asyncio.get_running_loop()

        def on_log(msg: str) -> None:
            self._emit_log(job_id, msg)

        plan = prepare_export(
            checkpoint=checkpoint,
            output_dir=output_dir,
            formats=payload.get("formats"),
            profile=payload.get("profile"),
            gguf_quantizations=payload.get("gguf_quantizations"),
            hub_repo=payload.get("hub_repo"),
            hub_token=payload.get("hub_token"),
            hub_metadata=hub_metadata,
            on_log=on_log,
        )

        if plan.precheck and not plan.precheck.ok:
            from seiso.export.hub_precheck import assert_hub_precheck_ok

            assert_hub_precheck_ok(plan.precheck)

        for warning in plan.warnings:
            self._emit_log(job_id, f"Warning: {warning}")

        results = await loop.run_in_executor(
            None,
            lambda: run_export_plan(
                plan,
                hub_token=payload.get("hub_token"),
                sandbox_root=self.sandbox_root,
                on_log=on_log,
            ),
        )
        paths = {k: str(v) for k, v in results.items()}
        ollama_commands = build_ollama_create_commands(paths, model_name=checkpoint.name)
        for command in ollama_commands:
            self._emit_log(job_id, f"Ollama: {command}")
        self._emit_log(job_id, "Export complete")
        return {
            "outputs": paths,
            "profile": plan.profile,
            "checkpoint_kind": plan.checkpoint_kind,
            "ollama_commands": ollama_commands,
        }
