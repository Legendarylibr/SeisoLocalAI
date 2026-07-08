"""RL quantization job persistence."""

from __future__ import annotations

import json

from forge.db.stores.constants import _RL_QUANT_LIST_COLUMNS, now_iso


class RLQuantMixin:
    async def create_rl_quant_job(
        self, user_id: str, config: dict, job_id: str | None = None
    ) -> dict:
        return await self._create_config_job(
            "rl_quant_jobs", user_id, config, job_id=job_id
        )

    async def get_rl_quant_job(self, job_id: str, user_id: str) -> dict | None:
        return await self._get_config_job("rl_quant_jobs", job_id, user_id)

    async def update_rl_quant_job_status(
        self,
        job_id: str,
        status: str,
        *,
        output_dir: str | None = None,
        recommendation_path: str | None = None,
        recommendation_json: dict | None = None,
        gguf_quants: list[str] | None = None,
        error_text: str | None = None,
    ) -> None:
        now = now_iso()
        async with self._conn() as conn:
            await conn.execute(
                """UPDATE rl_quant_jobs SET status = ?, updated_at = ?,
                   output_dir = COALESCE(?, output_dir),
                   recommendation_path = COALESCE(?, recommendation_path),
                   recommendation_json = COALESCE(?, recommendation_json),
                   gguf_quants_json = COALESCE(?, gguf_quants_json),
                   error_text = COALESCE(?, error_text)
                   WHERE id = ?""",
                (
                    status,
                    now,
                    output_dir,
                    recommendation_path,
                    (
                        json.dumps(recommendation_json)
                        if recommendation_json is not None
                        else None
                    ),
                    json.dumps(gguf_quants) if gguf_quants is not None else None,
                    error_text,
                    job_id,
                ),
            )
            await conn.commit()

    async def list_rl_quant_jobs(self, user_id: str) -> list[dict]:
        return await self._list_config_jobs(
            "rl_quant_jobs", user_id, columns=_RL_QUANT_LIST_COLUMNS
        )
