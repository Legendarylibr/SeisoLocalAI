"""Training job persistence."""

from __future__ import annotations

import json
import uuid

from forge.db.stores.constants import _TRAINING_LIST_COLUMNS, column_list, now_iso


class TrainingMixin:
    async def create_training_job(
        self,
        user_id: str,
        config: dict,
        project_id: str | None = None,
        job_id: str | None = None,
    ) -> dict:
        jid = job_id or str(uuid.uuid4())
        now = now_iso()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO training_jobs
                   (id, user_id, project_id, status, config_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (jid, user_id, project_id, "pending", json.dumps(config), now, now),
            )
            await conn.commit()
        return {"id": jid, "status": "pending", "config": config, "created_at": now}

    async def get_training_job(self, job_id: str, user_id: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM training_jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        *,
        checkpoint_path: str | None = None,
        metrics: dict | None = None,
        error_text: str | None = None,
        user_id: str | None = None,
    ) -> None:
        now = now_iso()
        owner_clause = " AND user_id = ?" if user_id else ""
        async with self._conn() as conn:
            if checkpoint_path or metrics is not None or error_text is not None:
                query = f"""UPDATE training_jobs SET status = ?, updated_at = ?,
                       checkpoint_path = COALESCE(?, checkpoint_path),
                       metrics_json = COALESCE(?, metrics_json),
                       error_text = COALESCE(?, error_text)
                       WHERE id = ?{owner_clause}"""  # nosec B608
                await conn.execute(
                    query,
                    (
                        status,
                        now,
                        checkpoint_path,
                        json.dumps(metrics or {}) if metrics is not None else None,
                        error_text,
                        job_id,
                        *([user_id] if user_id else []),
                    ),
                )
            else:
                await conn.execute(
                    f"UPDATE training_jobs SET status = ?, updated_at = ? WHERE id = ?{owner_clause}",  # nosec B608
                    (status, now, job_id, *([user_id] if user_id else [])),
                )
            await conn.commit()

    async def update_training_metrics(
        self, job_id: str, metrics: dict, *, user_id: str | None = None
    ) -> None:
        now = now_iso()
        owner_clause = " AND user_id = ?" if user_id else ""
        async with self._conn() as conn:
            await conn.execute(
                f"UPDATE training_jobs SET metrics_json = ?, updated_at = ? WHERE id = ?{owner_clause}",  # nosec B608
                (json.dumps(metrics), now, job_id, *([user_id] if user_id else [])),
            )
            await conn.commit()

    async def list_training_jobs(self, user_id: str) -> list[dict]:
        cols = column_list(_TRAINING_LIST_COLUMNS)
        async with (
            self._conn() as conn,
            conn.execute(
                f"SELECT {cols} FROM training_jobs WHERE user_id = ? ORDER BY created_at DESC",  # nosec B608
                (user_id,),
            ) as cur,
        ):
            return [dict(r) for r in await cur.fetchall()]
