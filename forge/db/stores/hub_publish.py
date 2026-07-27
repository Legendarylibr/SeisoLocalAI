"""Hub publish job persistence."""

from __future__ import annotations

import json
import uuid

from forge.db.stores.constants import _HUB_PUBLISH_LIST_COLUMNS, column_list, now_iso


class HubPublishMixin:
    async def create_hub_publish_job(
        self, user_id: str, config: dict, job_id: str | None = None
    ) -> dict:
        jid = job_id or str(uuid.uuid4())
        now = now_iso()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO hub_publish_jobs
                   (id, user_id, status, config_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (jid, user_id, "pending", self._enc(json.dumps(config)), now, now),
            )
            await conn.commit()
        return {"id": jid, "status": "pending", "config": config, "created_at": now}

    async def get_hub_publish_job(self, job_id: str, user_id: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM hub_publish_jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return self._decrypt_row("hub_publish_jobs", dict(row)) if row else None

    async def update_hub_publish_job_status(
        self,
        job_id: str,
        status: str,
        *,
        user_id: str | None = None,
        result: dict | None = None,
        error_text: str | None = None,
    ) -> None:
        now = now_iso()
        owner_clause = " AND user_id = ?" if user_id is not None else ""
        async with self._conn() as conn:
            if result is not None or error_text is not None:
                await conn.execute(
                    f"""UPDATE hub_publish_jobs SET status = ?, updated_at = ?,
                       result_json = COALESCE(?, result_json),
                       error_text = COALESCE(?, error_text)
                       WHERE id = ?{owner_clause}""",  # nosec B608
                    (
                        status,
                        now,
                        json.dumps(result) if result is not None else None,
                        error_text,
                        job_id,
                        *((user_id,) if user_id is not None else ()),
                    ),
                )
            else:
                await conn.execute(
                    f"UPDATE hub_publish_jobs SET status = ?, updated_at = ? WHERE id = ?{owner_clause}",  # nosec B608
                    (status, now, job_id, *((user_id,) if user_id is not None else ())),
                )
            await conn.commit()

    async def list_hub_publish_jobs(self, user_id: str) -> list[dict]:
        cols = column_list(_HUB_PUBLISH_LIST_COLUMNS)
        async with (
            self._conn() as conn,
            conn.execute(
                f"SELECT {cols} FROM hub_publish_jobs WHERE user_id = ? ORDER BY created_at DESC",  # nosec B608
                (user_id,),
            ) as cur,
        ):
            return [dict(r) for r in await cur.fetchall()]
