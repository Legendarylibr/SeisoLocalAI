"""Local model inventory persistence."""

from __future__ import annotations

import json
import uuid
from typing import Any

from forge.db.stores.constants import _UPSERT_MODEL_SQL, now_iso


class ModelsMixin:
    async def list_models(self, user_id: str | None = None) -> list[dict]:
        async with self._conn() as conn:
            if user_id:
                async with conn.execute(
                    """SELECT * FROM local_models
                       WHERE user_id = ? OR user_id IS NULL
                       ORDER BY created_at DESC, user_id IS NULL ASC""",
                    (user_id,),
                ) as cur:
                    rows = await cur.fetchall()
                    return [dict(r) for r in rows]

            async with conn.execute("SELECT * FROM local_models ORDER BY created_at DESC") as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_model(self, model_id: str, user_id: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM local_models WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
                (model_id, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_model_by_path(self, user_id: str, path: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM local_models WHERE path = ? AND (user_id = ? OR user_id IS NULL) ORDER BY created_at DESC LIMIT 1",
                (path, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_model_by_name(self, user_id: str, name: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM local_models WHERE name = ? AND (user_id = ? OR user_id IS NULL) ORDER BY created_at DESC LIMIT 1",
                (name, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def add_model(self, **fields: Any) -> dict:
        mid = str(uuid.uuid4())
        now = now_iso()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO local_models
                   (id, user_id, name, path, source, format, size_bytes, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mid,
                    fields.get("user_id"),
                    fields["name"],
                    fields["path"],
                    fields.get("source"),
                    fields.get("format"),
                    fields.get("size_bytes", 0),
                    json.dumps(fields.get("metadata", {})),
                    now,
                ),
            )
            await conn.commit()
        return {"id": mid, **fields, "created_at": now}

    async def get_model_by_source(self, user_id: str, source: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM local_models WHERE user_id = ? AND source = ? ORDER BY created_at DESC LIMIT 1",
                (user_id, source),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_model_by_metadata_repo_id(self, user_id: str, repo_id: str) -> dict | None:
        """Lookup by metadata_json.repo_id without scanning the full inventory."""
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM local_models WHERE user_id = ? "
                "AND json_extract(metadata_json, '$.repo_id') = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id, repo_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def upsert_model(self, user_id: str, source: str, **fields: Any) -> dict:
        if source is None:
            # ON CONFLICT(user_id, source) no-ops on NULL (SQLite NULLs are
            # distinct), which would insert duplicate rows on every call.
            raise ValueError("upsert_model source must not be None")
        mid = str(uuid.uuid4())
        now = now_iso()
        async with self._conn() as conn:
            await conn.execute(
                _UPSERT_MODEL_SQL,
                (
                    mid,
                    user_id,
                    fields["name"],
                    fields["path"],
                    source,
                    fields.get("format"),
                    fields.get("size_bytes", 0),
                    json.dumps(fields.get("metadata", {})),
                    now,
                ),
            )
            await conn.commit()
        row = await self.get_model_by_source(user_id, source)
        if row is None:
            raise RuntimeError("upsert_model failed to persist row")
        return row

    async def upsert_models(self, user_id: str, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        for record in records:
            if record.get("source") is None:
                # NULL source defeats ON CONFLICT(user_id, source) (see upsert_model).
                raise ValueError("upsert_models records must have a non-None source")
        now = now_iso()
        rows = [
            (
                str(uuid.uuid4()),
                user_id,
                record["name"],
                record["path"],
                record["source"],
                record.get("format"),
                record.get("size_bytes", 0),
                json.dumps(record.get("metadata", {})),
                now,
            )
            for record in records
        ]
        async with self._conn() as conn:
            await conn.executemany(_UPSERT_MODEL_SQL, rows)
            await conn.commit()
        return len(rows)
