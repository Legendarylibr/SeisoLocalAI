"""Model export job persistence."""

from __future__ import annotations

import json
import uuid

from forge.db.stores.constants import _EXPORT_LIST_COLUMNS, column_list, now_iso


class ExportMixin:
    async def create_export_job(
        self, user_id: str, config: dict, job_id: str | None = None
    ) -> dict:
        jid = job_id or str(uuid.uuid4())
        now = now_iso()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO export_jobs
                   (id, user_id, status, config_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (jid, user_id, "pending", json.dumps(config), now, now),
            )
            await conn.commit()
        return {"id": jid, "status": "pending", "config": config, "created_at": now}

    async def get_export_job(self, job_id: str, user_id: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM export_jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_export_job_status(
        self,
        job_id: str,
        status: str,
        *,
        user_id: str | None = None,
        output_paths: dict | None = None,
        error_text: str | None = None,
    ) -> None:
        now = now_iso()
        owner_clause = " AND user_id = ?" if user_id is not None else ""
        async with self._conn() as conn:
            if output_paths is not None or error_text is not None:
                await conn.execute(
                    f"""UPDATE export_jobs SET status = ?, updated_at = ?,
                       output_paths_json = COALESCE(?, output_paths_json),
                       error_text = COALESCE(?, error_text)
                       WHERE id = ?{owner_clause}""",  # nosec B608
                    (
                        status,
                        now,
                        json.dumps(output_paths) if output_paths is not None else None,
                        error_text,
                        job_id,
                        *((user_id,) if user_id is not None else ()),
                    ),
                )
            else:
                await conn.execute(
                    f"UPDATE export_jobs SET status = ?, updated_at = ? WHERE id = ?{owner_clause}",  # nosec B608
                    (status, now, job_id, *((user_id,) if user_id is not None else ())),
                )
            await conn.commit()

    async def list_export_jobs(self, user_id: str) -> list[dict]:
        cols = column_list(_EXPORT_LIST_COLUMNS)
        async with (
            self._conn() as conn,
            conn.execute(
                f"SELECT {cols} FROM export_jobs WHERE user_id = ? ORDER BY created_at DESC",  # nosec B608
                (user_id,),
            ) as cur,
        ):
            return [dict(r) for r in await cur.fetchall()]
