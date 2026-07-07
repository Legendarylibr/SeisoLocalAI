"""Export job orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path
from seiso.export.hub_precheck import hub_precheck_from_dict
from seiso.export.model_card import HubModelMetadata
from seiso.export.pipeline import prepare_export, run_export_plan
from seiso.security import SecurityError, assert_within


class ExportOrchestrator(Orchestrator):
    kind = "export"
    resource_key = "gpu"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        from forge.services.memory_release import (
            prepare_for_gpu_task,
            release_after_task,
        )

        user_id = payload.get("user_id")
        if not user_id:
            raise PermissionError("user_id required for export")
        try:
            checkpoint = assert_user_path(
                self.sandbox_root, user_id, payload["checkpoint"]
            )
        except SecurityError as exc:
            raise PermissionError(str(exc)) from exc

        prepare_for_gpu_task(
            task="export",
            job_id=job_id,
            log=lambda msg: self._emit_log(job_id, msg),
        )

        from seiso.models.hf_env import configure_hf_hub_cache

        configure_hf_hub_cache(self.sandbox_root)

        hub_meta_raw = payload.get("hub_metadata")
        hub_metadata = HubModelMetadata(**hub_meta_raw) if hub_meta_raw else None
        user_exports = (self.sandbox_root / "exports" / user_id).resolve()
        default_output = user_exports / job_id
        raw_output = Path(payload.get("output_dir", default_output))
        if not raw_output.is_absolute():
            raw_output = user_exports / raw_output
        try:
            output_dir = assert_within(user_exports, raw_output)
        except SecurityError as exc:
            raise PermissionError(str(exc)) from exc

        self._emit_log(job_id, f"Exporting checkpoint: {checkpoint.name}")

        loop = asyncio.get_running_loop()

        def on_log(msg: str) -> None:
            loop.call_soon_threadsafe(self._emit_log, job_id, msg)

        try:
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
                hub_precheck=hub_precheck_from_dict(payload.get("hub_precheck")),
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
        finally:
            release_after_task(
                reason="export complete",
                log=lambda msg: self._emit_log(job_id, msg),
                job_id=job_id,
            )

        paths = {k: str(v) for k, v in results.items()}
        self._emit_log(job_id, "Export complete")
        return {
            "outputs": paths,
            "profile": plan.profile,
            "checkpoint_kind": plan.checkpoint_kind,
        }
