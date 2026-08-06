"""Durable job event persistence for restart-safe streams."""

from __future__ import annotations

import json
import uuid
from typing import Any

from forge.db.stores.constants import now_iso


class JobEventsMixin:
    async def append_job_event(
        self,
        *,
        job_id: str,
        user_id: str,
        kind: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Append a durable log/metric/status event for restart-safe streams."""
        now = now_iso()
        event_id = str(uuid.uuid4())
        async with self._conn() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            cur = await conn.execute(
                """SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                   FROM job_events WHERE job_id = ?""",
                (job_id,),
            )
            row = await cur.fetchone()
            sequence = int(row["next_sequence"]) if row else 1
            await conn.execute(
                """INSERT INTO job_events
                   (id, job_id, user_id, kind, event_type, payload_json, sequence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    job_id,
                    user_id,
                    kind,
                    event_type,
                    json.dumps(payload, default=str),
                    sequence,
                    now,
                ),
            )
            await conn.commit()
        return {
            "id": event_id,
            "job_id": job_id,
            "user_id": user_id,
            "kind": kind,
            "event_type": event_type,
            "payload": payload,
            "sequence": sequence,
            "created_at": now,
        }

    async def list_job_events(
        self,
        job_id: str,
        user_id: str,
        *,
        event_types: tuple[str, ...] | None = None,
        after_sequence: int = 0,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Return decoded durable job events in sequence order."""
        limit = max(1, min(int(limit), 10000))
        params: list[Any] = [job_id, user_id, after_sequence]
        type_clause = ""
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            type_clause = f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        params.append(limit)
        query = f"""SELECT * FROM job_events
                   WHERE job_id = ? AND user_id = ? AND sequence > ?{type_clause}
                   ORDER BY sequence ASC
                   LIMIT ?"""  # nosec B608
        async with (
            self._conn() as conn,
            conn.execute(query, params) as cur,
        ):
            rows = [dict(r) for r in await cur.fetchall()]
        for row in rows:
            try:
                row["payload"] = json.loads(row.get("payload_json") or "{}")
            except json.JSONDecodeError:
                row["payload"] = {}
        return rows

    async def prune_job_events(self, job_id: str, user_id: str, *, keep_last: int = 5000) -> int:
        """Keep the newest N events for a job and delete older rows."""
        keep_last = max(1, int(keep_last))
        async with self._conn() as conn:
            cur = await conn.execute(
                """DELETE FROM job_events
                   WHERE job_id = ? AND user_id = ?
                     AND sequence NOT IN (
                       SELECT sequence FROM job_events
                       WHERE job_id = ? AND user_id = ?
                       ORDER BY sequence DESC
                       LIMIT ?
                     )""",
                (job_id, user_id, job_id, user_id, keep_last),
            )
            await conn.commit()
            return int(cur.rowcount or 0)
