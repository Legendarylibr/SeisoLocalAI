"""Background Hugging Face publish jobs for large GGUF uploads."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from seiso.export.formats import publish_folder_to_hub
from seiso.export.model_card import HubModelMetadata
from seiso.models.hf_env import configure_hf_hub_cache


class HubPublishOrchestrator(Orchestrator):
    kind = "hub_publish"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        configure_hf_hub_cache(self.sandbox_root)
        loop = asyncio.get_running_loop()

        def on_log(msg: str) -> None:
            loop.call_soon_threadsafe(self._emit_log, job_id, msg)

        from forge.services.executors import IO_EXECUTOR

        rec = self.get_job(job_id)
        user_id = str(payload.get("user_id") or (rec.user_id if rec else "") or "")
        return await loop.run_in_executor(
            IO_EXECUTOR, lambda: self._run_publish(payload, on_log, user_id=user_id)
        )

    def _run_publish(
        self,
        payload: dict[str, Any],
        on_log: Any,
        *,
        user_id: str,
    ) -> dict[str, str]:
        from forge.services.user_paths import assert_user_path

        folder = Path(payload["folder"])
        if not user_id:
            raise ValueError("user_id is required for hub publish")
        assert_user_path(self.sandbox_root, user_id, folder)
        repo_id = payload["repo_id"]
        token = payload["token"]
        meta_raw = payload.get("metadata") or {}
        meta = HubModelMetadata(**meta_raw)
        quantizations = payload.get("quantizations")

        on_log(f"Publishing {folder} to https://huggingface.co/{repo_id}")
        publish_folder_to_hub(
            folder,
            repo_id=repo_id,
            token=token,
            metadata=meta,
            quantizations=quantizations,
            on_log=on_log,
            skip_precheck=True,
            data_dir=self.sandbox_root,
        )
        on_log(f"Published to https://huggingface.co/{repo_id}")
        return {"repo_id": repo_id, "path": str(folder)}
