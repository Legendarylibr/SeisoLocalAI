"""Connection, crypto, schema, and shared config-job helpers."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from forge.db.crypto import decrypt_field, encrypt_field
from forge.db.stores.constants import (
    _JOB_ERROR_TABLES,
    _STAGE_PIPELINE_LIST_COLUMNS,
    ENCRYPTED_COLUMNS,
    SCHEMA,
    column_list,
    config_job_table,
    now_iso,
)


class DatabaseCryptoError(RuntimeError):
    """Raised when encrypted database fields cannot be decrypted safely."""


class DatabaseCore:
    """Shared connection, crypto, and schema lifecycle for domain store mixins."""

    def __init__(
        self,
        path: Path,
        *,
        encryption_key: bytes,
        ephemeral: bool = True,
    ) -> None:
        self.path = path
        self._encryption_key = encryption_key
        self._ephemeral = ephemeral
        self._initialized = False
        self._conn_holder: aiosqlite.Connection | None = None
        # Single shared aiosqlite connection: serialize concurrent users so
        # BEGIN IMMEDIATE / commit pairs cannot nest across fire-and-forget tasks.
        self._conn_lock = asyncio.Lock()
        if ephemeral:
            self._dsn = f"file:seiso_{uuid.uuid4().hex}?mode=memory&cache=shared"
        else:
            self._dsn = str(path)

    def _enc(self, value: str) -> str:
        return encrypt_field(value, self._encryption_key)

    def _dec(self, value: str) -> str:
        try:
            return decrypt_field(value, self._encryption_key)
        except Exception as exc:
            raise DatabaseCryptoError(
                "Encrypted database field could not be decrypted"
            ) from exc

    def _decrypt_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for column in ENCRYPTED_COLUMNS.get(table, ()):
            if column in out and out[column] is not None:
                out[column] = self._dec(str(out[column]))
        return out

    async def _configure(self, conn: aiosqlite.Connection) -> None:
        await conn.execute("PRAGMA busy_timeout = 5000")
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA temp_store = MEMORY")
        if not self._ephemeral:
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA synchronous = NORMAL")
            await conn.execute("PRAGMA cache_size = -64000")
            await conn.execute("PRAGMA mmap_size = 268435456")

    async def _migrate_schema(self, conn: aiosqlite.Connection) -> None:
        # Drop schema-only leftovers that never had writers (F4-03/04/05).
        for dead_table in ("recipe_jobs", "knowledge_bases", "projects"):
            await conn.execute(f"DROP TABLE IF EXISTS {dead_table}")  # nosec B608
        for table in _JOB_ERROR_TABLES:
            async with conn.execute(f"PRAGMA table_info({table})") as cur:
                cols = {row[1] for row in await cur.fetchall()}
            if "error_text" not in cols:
                await conn.execute(f"ALTER TABLE {table} ADD COLUMN error_text TEXT")
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_models_user_source ON local_models(user_id, source)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_user_created ON local_models(user_id, created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_user_path ON local_models(user_id, path)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_user_name ON local_models(user_id, name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_threads_user_updated ON chat_threads(user_id, updated_at DESC)"
        )
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON training_jobs(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_thread_created ON chat_messages(thread_id, created_at ASC)",
            "CREATE INDEX IF NOT EXISTS idx_export_jobs_user_created ON export_jobs(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_hub_publish_jobs_user_created ON hub_publish_jobs(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_compress_jobs_user_created ON compress_jobs(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_distill_rl_jobs_user_created ON distill_rl_jobs(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_rl_quant_jobs_user_created ON rl_quant_jobs(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_providers_user_created ON providers(user_id, created_at DESC)",
        ):
            await conn.execute(statement)

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn_holder is None:
            if self._ephemeral:
                conn = await aiosqlite.connect(self._dsn, uri=True)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = await aiosqlite.connect(self._dsn)
            conn.row_factory = aiosqlite.Row
            await self._configure(conn)
            if not self._initialized:
                await conn.executescript(SCHEMA)
                await conn.commit()
                await self._migrate_schema(conn)
                await conn.commit()
                self._initialized = True
            self._conn_holder = conn
        return self._conn_holder

    async def close(self) -> None:
        if self._conn_holder is not None:
            await self._conn_holder.close()
            self._conn_holder = None
        self._initialized = False

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._conn_lock:
            yield await self._ensure_conn()

    async def _create_config_job(
        self, table: str, user_id: str, config: dict, job_id: str | None = None
    ) -> dict:
        table = config_job_table(table)
        jid = job_id or str(uuid.uuid4())
        now = now_iso()
        async with self._conn() as conn:
            query = f"""INSERT INTO {table}
                   (id, user_id, status, config_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)"""  # nosec B608
            await conn.execute(
                query,
                (jid, user_id, "pending", json.dumps(config), now, now),
            )
            await conn.commit()
        return {"id": jid, "status": "pending", "config": config, "created_at": now}

    async def _get_config_job(
        self, table: str, job_id: str, user_id: str
    ) -> dict | None:
        table = config_job_table(table)
        query = f"SELECT * FROM {table} WHERE id = ? AND user_id = ?"  # nosec B608
        async with (
            self._conn() as conn,
            conn.execute(
                query,
                (job_id, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def _list_config_jobs(
        self,
        table: str,
        user_id: str,
        *,
        columns: tuple[str, ...] = _STAGE_PIPELINE_LIST_COLUMNS,
    ) -> list[dict]:
        table = config_job_table(table)
        cols = column_list(columns)
        query = f"SELECT {cols} FROM {table} WHERE user_id = ? ORDER BY created_at DESC"  # nosec B608
        async with (
            self._conn() as conn,
            conn.execute(
                query,
                (user_id,),
            ) as cur,
        ):
            return [dict(r) for r in await cur.fetchall()]

    async def _update_stage_pipeline_job_status(
        self,
        table: str,
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
        table = config_job_table(table)
        now = now_iso()
        owner_clause = " AND user_id = ?" if user_id is not None else ""
        async with self._conn() as conn:
            query = f"""UPDATE {table} SET status = ?, updated_at = ?,
                   output_dir = COALESCE(?, output_dir),
                   run_dir = COALESCE(?, run_dir),
                   model_dir = COALESCE(?, model_dir),
                   stages_json = COALESCE(?, stages_json),
                   stage_results_json = COALESCE(?, stage_results_json),
                   error_text = COALESCE(?, error_text)
                   WHERE id = ?{owner_clause}"""  # nosec B608
            params: list[Any] = [
                status,
                now,
                output_dir,
                run_dir,
                model_dir,
                json.dumps(stages) if stages is not None else None,
                json.dumps(stage_results) if stage_results is not None else None,
                error_text,
                job_id,
            ]
            if user_id is not None:
                params.append(user_id)
            await conn.execute(query, tuple(params))
            await conn.commit()
