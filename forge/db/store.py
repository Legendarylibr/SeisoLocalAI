"""SQLite persistence layer."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from forge.db.crypto import decrypt_field, encrypt_field

ENCRYPTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "chat_messages": ("content", "metadata_json"),
    "providers": ("config_json",),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS local_models (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    source TEXT,
    format TEXT,
    size_bytes INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    checkpoint_path TEXT,
    metrics_json TEXT DEFAULT '{}',
    error_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS export_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    output_paths_json TEXT DEFAULT '{}',
    error_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rl_quant_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    output_dir TEXT,
    recommendation_path TEXT,
    recommendation_json TEXT DEFAULT '{}',
    gguf_quants_json TEXT DEFAULT '[]',
    error_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS compress_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    output_dir TEXT,
    run_dir TEXT,
    model_dir TEXT,
    stages_json TEXT DEFAULT '[]',
    stage_results_json TEXT DEFAULT '{}',
    error_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_compress_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    output_dir TEXT,
    run_dir TEXT,
    model_dir TEXT,
    stages_json TEXT DEFAULT '[]',
    stage_results_json TEXT DEFAULT '{}',
    error_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    title TEXT NOT NULL,
    model_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES chat_threads(id)
);

CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recipe_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    recipe_json TEXT NOT NULL,
    output_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_user ON local_models(user_id);
CREATE INDEX IF NOT EXISTS idx_threads_user ON chat_threads(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_export_jobs_user ON export_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_compress_jobs_user ON compress_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_image_compress_jobs_user ON image_compress_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_rl_quant_jobs_user ON rl_quant_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_recipe_jobs_user ON recipe_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_providers_user ON providers(user_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_user ON knowledge_bases(user_id);
"""

_JOB_ERROR_TABLES = (
    "training_jobs",
    "export_jobs",
    "rl_quant_jobs",
    "compress_jobs",
    "image_compress_jobs",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
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
        if ephemeral:
            self._dsn = f"file:seiso_{uuid.uuid4().hex}?mode=memory&cache=shared"
        else:
            self._dsn = str(path)

    def _enc(self, value: str) -> str:
        return encrypt_field(value, self._encryption_key)

    def _dec(self, value: str) -> str:
        return decrypt_field(value, self._encryption_key)

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

    async def _migrate_schema(self, conn: aiosqlite.Connection) -> None:
        for table in _JOB_ERROR_TABLES:
            async with conn.execute(f"PRAGMA table_info({table})") as cur:
                cols = {row[1] for row in await cur.fetchall()}
            if "error_text" not in cols:
                await conn.execute(f"ALTER TABLE {table} ADD COLUMN error_text TEXT")

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
        yield await self._ensure_conn()

    async def user_count(self) -> int:
        async with self._conn() as conn, conn.execute("SELECT COUNT(*) AS c FROM users") as cur:
            row = await cur.fetchone()
            return int(row["c"]) if row else 0

    async def create_first_user(
        self,
        password_hash: str,
        display_name: str,
        *,
        email: str | None = None,
    ) -> dict:
        """Atomically create the sole local user (registration is single-tenant)."""
        uid = str(uuid.uuid4())
        now = _now()
        normalized_name = display_name.strip()
        resolved_email = (email or f"{uid}@local.seiso").lower()
        async with self._conn() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            cur = await conn.execute("SELECT COUNT(*) AS c FROM users")
            row = await cur.fetchone()
            if row and int(row["c"]) > 0:
                await conn.execute("ROLLBACK")
                raise ValueError("Registration closed — user already exists")
            await conn.execute(
                "INSERT INTO users (id, email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (uid, resolved_email, password_hash, normalized_name, now),
            )
            await conn.commit()
        return {
            "id": uid,
            "email": resolved_email,
            "display_name": normalized_name,
            "created_at": now,
        }

    async def create_user(
        self,
        password_hash: str,
        display_name: str,
        *,
        email: str | None = None,
    ) -> dict:
        uid = str(uuid.uuid4())
        now = _now()
        normalized_name = display_name.strip()
        resolved_email = (email or f"{uid}@local.seiso").lower()
        async with self._conn() as conn:
            await conn.execute(
                "INSERT INTO users (id, email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (uid, resolved_email, password_hash, normalized_name, now),
            )
            await conn.commit()
        return {
            "id": uid,
            "email": resolved_email,
            "display_name": normalized_name,
            "created_at": now,
        }

    async def get_user_by_display_name(self, display_name: str) -> dict | None:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM users WHERE lower(display_name) = ?",
            (display_name.strip().lower(),),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_sole_user(self) -> dict | None:
        """Return the single local user, if any (Forge allows one account per instance)."""
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM users ORDER BY created_at ASC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_user_by_email(self, email: str) -> dict | None:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower(),)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_user_by_id(self, user_id: str) -> dict | None:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_models(self, user_id: str | None = None) -> list[dict]:
        async with self._conn() as conn:
            if user_id:
                q = "SELECT * FROM local_models WHERE user_id = ? OR user_id IS NULL ORDER BY created_at DESC"
                params: tuple[Any, ...] = (user_id,)
            else:
                q = "SELECT * FROM local_models ORDER BY created_at DESC"
                params = ()
            async with conn.execute(q, params) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_model(self, model_id: str, user_id: str) -> dict | None:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM local_models WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (model_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def add_model(self, **fields: Any) -> dict:
        mid = str(uuid.uuid4())
        now = _now()
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
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM local_models WHERE user_id = ? AND source = ? ORDER BY created_at DESC LIMIT 1",
            (user_id, source),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def upsert_model(self, user_id: str, source: str, **fields: Any) -> dict:
        existing = await self.get_model_by_source(user_id, source)
        if not existing:
            return await self.add_model(user_id=user_id, source=source, **fields)

        async with self._conn() as conn:
            await conn.execute(
                """UPDATE local_models
                   SET name = ?, path = ?, format = ?, size_bytes = ?, metadata_json = ?
                   WHERE id = ?""",
                (
                    fields["name"],
                    fields["path"],
                    fields.get("format"),
                    fields.get("size_bytes", 0),
                    json.dumps(fields.get("metadata", {})),
                    existing["id"],
                ),
            )
            await conn.commit()
        return {**existing, **fields, "id": existing["id"], "user_id": user_id, "source": source}

    async def create_training_job(
        self,
        user_id: str,
        config: dict,
        project_id: str | None = None,
        job_id: str | None = None,
    ) -> dict:
        jid = job_id or str(uuid.uuid4())
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO training_jobs
                   (id, user_id, project_id, status, config_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (jid, user_id, project_id, "pending", json.dumps(config), now, now),
            )
            await conn.commit()
        return {"id": jid, "status": "pending", "config": config, "created_at": now}

    async def get_thread_for_user(self, thread_id: str, user_id: str) -> dict | None:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM chat_threads WHERE id = ? AND user_id = ?",
            (thread_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_training_job(self, job_id: str, user_id: str) -> dict | None:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM training_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ) as cur:
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
    ) -> None:
        now = _now()
        async with self._conn() as conn:
            if checkpoint_path or metrics is not None or error_text is not None:
                await conn.execute(
                    """UPDATE training_jobs SET status = ?, updated_at = ?,
                       checkpoint_path = COALESCE(?, checkpoint_path),
                       metrics_json = COALESCE(?, metrics_json),
                       error_text = COALESCE(?, error_text)
                       WHERE id = ?""",
                    (
                        status,
                        now,
                        checkpoint_path,
                        json.dumps(metrics or {}) if metrics is not None else None,
                        error_text,
                        job_id,
                    ),
                )
            else:
                await conn.execute(
                    "UPDATE training_jobs SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, job_id),
                )
            await conn.commit()

    async def update_training_metrics(self, job_id: str, metrics: dict) -> None:
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                "UPDATE training_jobs SET metrics_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metrics), now, job_id),
            )
            await conn.commit()

    async def list_training_jobs(self, user_id: str) -> list[dict]:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM training_jobs WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def create_thread(self, user_id: str, title: str, model_id: str | None = None) -> dict:
        tid = str(uuid.uuid4())
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO chat_threads (id, user_id, title, model_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (tid, user_id, title, model_id, now, now),
            )
            await conn.commit()
        return {"id": tid, "title": title, "model_id": model_id, "created_at": now}

    async def count_threads(self, user_id: str) -> int:
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM chat_threads WHERE user_id = ?",
                (user_id,),
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def list_threads(self, user_id: str) -> list[dict]:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM chat_threads WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def delete_thread(self, thread_id: str, user_id: str) -> bool:
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT id FROM chat_threads WHERE id = ? AND user_id = ?",
                (thread_id, user_id),
            )
            if not await cur.fetchone():
                return False
            await conn.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
            await conn.execute("DELETE FROM chat_threads WHERE id = ?", (thread_id,))
            await conn.commit()
            return True

    async def purge_user_chat(self, user_id: str) -> int:
        """Remove all chat threads and encrypted messages for a user (session end)."""
        async with self._conn() as conn:
            await conn.execute(
                "DELETE FROM chat_messages WHERE thread_id IN (SELECT id FROM chat_threads WHERE user_id = ?)",
                (user_id,),
            )
            cur = await conn.execute("DELETE FROM chat_threads WHERE user_id = ?", (user_id,))
            await conn.commit()
            return cur.rowcount

    async def add_message(self, thread_id: str, role: str, content: str, metadata: dict | None = None) -> dict:
        mid = str(uuid.uuid4())
        now = _now()
        enc_content = self._enc(content)
        enc_metadata = self._enc(json.dumps(metadata or {}))
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO chat_messages (id, thread_id, role, content, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (mid, thread_id, role, enc_content, enc_metadata, now),
            )
            await conn.execute(
                "UPDATE chat_threads SET updated_at = ? WHERE id = ?", (now, thread_id)
            )
            await conn.commit()
        return {"id": mid, "thread_id": thread_id, "role": role, "content": content, "created_at": now}

    async def get_messages(self, thread_id: str) -> list[dict]:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM chat_messages WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,),
        ) as cur:
            return [self._decrypt_row("chat_messages", dict(r)) for r in await cur.fetchall()]

    # --- Providers ---

    async def list_providers(self, user_id: str) -> list[dict]:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM providers WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ) as cur:
            return [self._decrypt_row("providers", dict(r)) for r in await cur.fetchall()]

    async def create_provider(
        self, user_id: str, name: str, provider_type: str, config: dict
    ) -> dict:
        pid = str(uuid.uuid4())
        now = _now()
        enc_config = self._enc(json.dumps(config))
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO providers (id, user_id, name, provider_type, config_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pid, user_id, name, provider_type, enc_config, now),
            )
            await conn.commit()
        return {"id": pid, "name": name, "provider_type": provider_type, "config": config, "created_at": now}

    async def get_provider(self, provider_id: str, user_id: str) -> dict | None:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return self._decrypt_row("providers", dict(row)) if row else None

    async def delete_provider(self, provider_id: str, user_id: str) -> bool:
        async with self._conn() as conn:
            cur = await conn.execute(
                "DELETE FROM providers WHERE id = ? AND user_id = ?",
                (provider_id, user_id),
            )
            await conn.commit()
            return cur.rowcount > 0

    # --- Export jobs ---

    async def create_export_job(self, user_id: str, config: dict, job_id: str | None = None) -> dict:
        jid = job_id or str(uuid.uuid4())
        now = _now()
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
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM export_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_export_job_status(
        self,
        job_id: str,
        status: str,
        *,
        output_paths: dict | None = None,
        error_text: str | None = None,
    ) -> None:
        now = _now()
        async with self._conn() as conn:
            if output_paths is not None or error_text is not None:
                await conn.execute(
                    """UPDATE export_jobs SET status = ?, updated_at = ?,
                       output_paths_json = COALESCE(?, output_paths_json),
                       error_text = COALESCE(?, error_text)
                       WHERE id = ?""",
                    (
                        status,
                        now,
                        json.dumps(output_paths) if output_paths is not None else None,
                        error_text,
                        job_id,
                    ),
                )
            else:
                await conn.execute(
                    "UPDATE export_jobs SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, job_id),
                )
            await conn.commit()

    async def list_export_jobs(self, user_id: str) -> list[dict]:
        async with self._conn() as conn, conn.execute(
            "SELECT * FROM export_jobs WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # --- Shared config-job helpers (RL quant, compress, image compress) ---

    async def _create_config_job(
        self, table: str, user_id: str, config: dict, job_id: str | None = None
    ) -> dict:
        jid = job_id or str(uuid.uuid4())
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                f"""INSERT INTO {table}
                   (id, user_id, status, config_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (jid, user_id, "pending", json.dumps(config), now, now),
            )
            await conn.commit()
        return {"id": jid, "status": "pending", "config": config, "created_at": now}

    async def _get_config_job(self, table: str, job_id: str, user_id: str) -> dict | None:
        async with self._conn() as conn, conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def _list_config_jobs(self, table: str, user_id: str) -> list[dict]:
        async with self._conn() as conn, conn.execute(
            f"SELECT * FROM {table} WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def _update_stage_pipeline_job_status(
        self,
        table: str,
        job_id: str,
        status: str,
        *,
        output_dir: str | None = None,
        run_dir: str | None = None,
        model_dir: str | None = None,
        stages: list[str] | None = None,
        stage_results: dict | None = None,
        error_text: str | None = None,
    ) -> None:
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                f"""UPDATE {table} SET status = ?, updated_at = ?,
                   output_dir = COALESCE(?, output_dir),
                   run_dir = COALESCE(?, run_dir),
                   model_dir = COALESCE(?, model_dir),
                   stages_json = COALESCE(?, stages_json),
                   stage_results_json = COALESCE(?, stage_results_json),
                   error_text = COALESCE(?, error_text)
                   WHERE id = ?""",
                (
                    status,
                    now,
                    output_dir,
                    run_dir,
                    model_dir,
                    json.dumps(stages) if stages is not None else None,
                    json.dumps(stage_results) if stage_results is not None else None,
                    error_text,
                    job_id,
                ),
            )
            await conn.commit()

    # --- RL quant jobs ---

    async def create_rl_quant_job(self, user_id: str, config: dict, job_id: str | None = None) -> dict:
        return await self._create_config_job("rl_quant_jobs", user_id, config, job_id=job_id)

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
        now = _now()
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
                    json.dumps(recommendation_json) if recommendation_json is not None else None,
                    json.dumps(gguf_quants) if gguf_quants is not None else None,
                    error_text,
                    job_id,
                ),
            )
            await conn.commit()

    async def list_rl_quant_jobs(self, user_id: str) -> list[dict]:
        return await self._list_config_jobs("rl_quant_jobs", user_id)

    # --- Compression jobs ---

    async def create_compress_job(self, user_id: str, config: dict, job_id: str | None = None) -> dict:
        return await self._create_config_job("compress_jobs", user_id, config, job_id=job_id)

    async def get_compress_job(self, job_id: str, user_id: str) -> dict | None:
        return await self._get_config_job("compress_jobs", job_id, user_id)

    async def update_compress_job_status(
        self,
        job_id: str,
        status: str,
        *,
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
            output_dir=output_dir,
            run_dir=run_dir,
            model_dir=model_dir,
            stages=stages,
            stage_results=stage_results,
            error_text=error_text,
        )

    async def list_compress_jobs(self, user_id: str) -> list[dict]:
        return await self._list_config_jobs("compress_jobs", user_id)

    # --- Image compression jobs ---

    async def create_image_compress_job(
        self, user_id: str, config: dict, job_id: str | None = None
    ) -> dict:
        return await self._create_config_job("image_compress_jobs", user_id, config, job_id=job_id)

    async def get_image_compress_job(self, job_id: str, user_id: str) -> dict | None:
        return await self._get_config_job("image_compress_jobs", job_id, user_id)

    async def update_image_compress_job_status(
        self,
        job_id: str,
        status: str,
        *,
        output_dir: str | None = None,
        run_dir: str | None = None,
        model_dir: str | None = None,
        stages: list[str] | None = None,
        stage_results: dict | None = None,
        error_text: str | None = None,
    ) -> None:
        await self._update_stage_pipeline_job_status(
            "image_compress_jobs",
            job_id,
            status,
            output_dir=output_dir,
            run_dir=run_dir,
            model_dir=model_dir,
            stages=stages,
            stage_results=stage_results,
            error_text=error_text,
        )

    async def list_image_compress_jobs(self, user_id: str) -> list[dict]:
        return await self._list_config_jobs("image_compress_jobs", user_id)
