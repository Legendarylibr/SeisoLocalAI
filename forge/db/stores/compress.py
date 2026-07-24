"""LLM compression job persistence."""

from __future__ import annotations


class CompressMixin:
    async def create_compress_job(
        self, user_id: str, config: dict, job_id: str | None = None
    ) -> dict:
        return await self._create_config_job(
            "compress_jobs", user_id, config, job_id=job_id
        )

    async def get_compress_job(self, job_id: str, user_id: str) -> dict | None:
        return await self._get_config_job("compress_jobs", job_id, user_id)

    async def update_compress_job_status(
        self,
        job_id: str,
        status: str,
        *,
        user_id: str | None = None,
        output_dir: str | None = None,
        run_dir: str | None = None,
        model_dir: str | None = None,
        stages: list[str] | None = None,
        stage_results: dict | None = None,
        error_text: str | None = None,
    ) -> None:
        await self._update_stage_pipeline_job_status(
            "compress_jobs",
            job_id,
            status,
            user_id=user_id,
            output_dir=output_dir,
            run_dir=run_dir,
            model_dir=model_dir,
            stages=stages,
            stage_results=stage_results,
            error_text=error_text,
        )

    async def list_compress_jobs(self, user_id: str) -> list[dict]:
        return await self._list_config_jobs("compress_jobs", user_id)
