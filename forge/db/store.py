"""SQLite persistence layer."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

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
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS export_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    output_paths_json TEXT DEFAULT '{}',
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

CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    command TEXT NOT NULL,
    args_json TEXT DEFAULT '[]',
    env_json TEXT DEFAULT '{}',
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_user ON local_models(user_id);
CREATE INDEX IF NOT EXISTS idx_threads_user ON chat_threads(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON training_jobs(user_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._initialized = False

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[aiosqlite.Connection]:
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        try:
            if not self._initialized:
                await conn.executescript(SCHEMA)
                await conn.commit()
                self._initialized = True
            yield conn
        finally:
            await conn.close()

    async def user_count(self) -> int:
        async with self._conn() as conn:
            async with conn.execute("SELECT COUNT(*) AS c FROM users") as cur:
                row = await cur.fetchone()
                return int(row["c"]) if row else 0

    async def create_user(self, email: str, password_hash: str, display_name: str | None) -> dict:
        uid = str(uuid.uuid4())
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                "INSERT INTO users (id, email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (uid, email.lower(), password_hash, display_name, now),
            )
            await conn.commit()
        return {"id": uid, "email": email.lower(), "display_name": display_name, "created_at": now}

    async def get_user_by_email(self, email: str) -> dict | None:
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower(),)
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def get_user_by_id(self, user_id: str) -> dict | None:
        async with self._conn() as conn:
            async with conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)) as cur:
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
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM chat_threads WHERE id = ? AND user_id = ?",
                (thread_id, user_id),
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def get_training_job(self, job_id: str, user_id: str) -> dict | None:
        async with self._conn() as conn:
            async with conn.execute(
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
    ) -> None:
        now = _now()
        async with self._conn() as conn:
            if checkpoint_path or metrics is not None:
                await conn.execute(
                    """UPDATE training_jobs SET status = ?, updated_at = ?,
                       checkpoint_path = COALESCE(?, checkpoint_path),
                       metrics_json = COALESCE(?, metrics_json)
                       WHERE id = ?""",
                    (status, now, checkpoint_path, json.dumps(metrics or {}), job_id),
                )
            else:
                await conn.execute(
                    "UPDATE training_jobs SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, job_id),
                )
            await conn.commit()

    async def list_training_jobs(self, user_id: str) -> list[dict]:
        async with self._conn() as conn:
            async with conn.execute(
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

    async def list_threads(self, user_id: str) -> list[dict]:
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM chat_threads WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def add_message(self, thread_id: str, role: str, content: str, metadata: dict | None = None) -> dict:
        mid = str(uuid.uuid4())
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO chat_messages (id, thread_id, role, content, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (mid, thread_id, role, content, json.dumps(metadata or {}), now),
            )
            await conn.execute(
                "UPDATE chat_threads SET updated_at = ? WHERE id = ?", (now, thread_id)
            )
            await conn.commit()
        return {"id": mid, "thread_id": thread_id, "role": role, "content": content, "created_at": now}

    async def get_messages(self, thread_id: str) -> list[dict]:
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM chat_messages WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    # --- Providers ---

    async def list_providers(self, user_id: str) -> list[dict]:
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM providers WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def create_provider(
        self, user_id: str, name: str, provider_type: str, config: dict
    ) -> dict:
        pid = str(uuid.uuid4())
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO providers (id, user_id, name, provider_type, config_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pid, user_id, name, provider_type, json.dumps(config), now),
            )
            await conn.commit()
        return {"id": pid, "name": name, "provider_type": provider_type, "config": config, "created_at": now}

    async def get_provider(self, provider_id: str, user_id: str) -> dict | None:
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM providers WHERE id = ? AND user_id = ?",
                (provider_id, user_id),
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def delete_provider(self, provider_id: str, user_id: str) -> bool:
        async with self._conn() as conn:
            cur = await conn.execute(
                "DELETE FROM providers WHERE id = ? AND user_id = ?",
                (provider_id, user_id),
            )
            await conn.commit()
            return cur.rowcount > 0

    # --- MCP servers ---

    async def list_mcp_servers(self, user_id: str) -> list[dict]:
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM mcp_servers WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def create_mcp_server(
        self,
        user_id: str,
        name: str,
        command: str,
        args: list | None = None,
        env: dict | None = None,
    ) -> dict:
        mid = str(uuid.uuid4())
        now = _now()
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO mcp_servers
                   (id, user_id, name, command, args_json, env_json, enabled, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
                (mid, user_id, name, command, json.dumps(args or []), json.dumps(env or {}), now),
            )
            await conn.commit()
        return {
            "id": mid,
            "name": name,
            "command": command,
            "args": args or [],
            "env": env or {},
            "enabled": True,
            "created_at": now,
        }

    async def get_mcp_server(self, server_id: str, user_id: str) -> dict | None:
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM mcp_servers WHERE id = ? AND user_id = ?",
                (server_id, user_id),
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def delete_mcp_server(self, server_id: str, user_id: str) -> bool:
        async with self._conn() as conn:
            cur = await conn.execute(
                "DELETE FROM mcp_servers WHERE id = ? AND user_id = ?",
                (server_id, user_id),
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
        async with self._conn() as conn:
            async with conn.execute(
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
    ) -> None:
        now = _now()
        async with self._conn() as conn:
            if output_paths is not None:
                await conn.execute(
                    """UPDATE export_jobs SET status = ?, updated_at = ?,
                       output_paths_json = ? WHERE id = ?""",
                    (status, now, json.dumps(output_paths), job_id),
                )
            else:
                await conn.execute(
                    "UPDATE export_jobs SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, job_id),
                )
            await conn.commit()

    async def list_export_jobs(self, user_id: str) -> list[dict]:
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM export_jobs WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]
