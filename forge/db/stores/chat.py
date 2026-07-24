"""Chat thread and message persistence."""

from __future__ import annotations

import json
import uuid

from forge.db.stores.constants import now_iso


class ChatMixin:
    async def get_thread_for_user(self, thread_id: str, user_id: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM chat_threads WHERE id = ? AND user_id = ?",
                (thread_id, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_thread_with_messages(
        self, thread_id: str, user_id: str
    ) -> tuple[dict | None, list[dict]]:
        """Load a thread and its messages in one connection acquisition."""
        async with self._conn() as conn:
            async with conn.execute(
                "SELECT * FROM chat_threads WHERE id = ? AND user_id = ?",
                (thread_id, user_id),
            ) as cur:
                thread_row = await cur.fetchone()
            if not thread_row:
                return None, []
            async with conn.execute(
                "SELECT * FROM chat_messages WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,),
            ) as cur:
                messages = [
                    self._message_with_metadata(
                        self._decrypt_row("chat_messages", dict(r))
                    )
                    for r in await cur.fetchall()
                ]
            return dict(thread_row), messages

    async def update_thread_model(
        self,
        thread_id: str,
        model_id: str | None,
        *,
        user_id: str | None = None,
    ) -> None:
        now = now_iso()
        owner_clause = " AND user_id = ?" if user_id is not None else ""
        async with self._conn() as conn:
            await conn.execute(
                f"UPDATE chat_threads SET model_id = ?, updated_at = ? WHERE id = ?{owner_clause}",  # nosec B608
                (model_id, now, thread_id, *((user_id,) if user_id is not None else ())),
            )
            await conn.commit()

    async def create_thread(
        self, user_id: str, title: str, model_id: str | None = None
    ) -> dict:
        tid = str(uuid.uuid4())
        now = now_iso()
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
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM chat_threads WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ) as cur,
        ):
            return [dict(r) for r in await cur.fetchall()]

    async def delete_thread(self, thread_id: str, user_id: str) -> bool:
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT id FROM chat_threads WHERE id = ? AND user_id = ?",
                (thread_id, user_id),
            )
            if not await cur.fetchone():
                return False
            await conn.execute(
                "DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,)
            )
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
            cur = await conn.execute(
                "DELETE FROM chat_threads WHERE user_id = ?", (user_id,)
            )
            await conn.commit()
            return cur.rowcount

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
        *,
        model_id: str | None = None,
        user_id: str | None = None,
    ) -> dict:
        mid = str(uuid.uuid4())
        now = now_iso()
        enc_content = self._enc(content)
        enc_metadata = self._enc(json.dumps(metadata or {}))
        async with self._conn() as conn:
            if user_id is not None:
                async with conn.execute(
                    "SELECT id FROM chat_threads WHERE id = ? AND user_id = ?",
                    (thread_id, user_id),
                ) as cur:
                    if await cur.fetchone() is None:
                        raise PermissionError(f"Thread {thread_id} not found for user")
            await conn.execute(
                """INSERT INTO chat_messages (id, thread_id, role, content, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (mid, thread_id, role, enc_content, enc_metadata, now),
            )
            owner_clause = " AND user_id = ?" if user_id is not None else ""
            owner_params = (user_id,) if user_id is not None else ()
            if model_id is not None:
                await conn.execute(
                    f"UPDATE chat_threads SET model_id = ?, updated_at = ? WHERE id = ?{owner_clause}",  # nosec B608
                    (model_id, now, thread_id, *owner_params),
                )
            else:
                await conn.execute(
                    f"UPDATE chat_threads SET updated_at = ? WHERE id = ?{owner_clause}",  # nosec B608
                    (now, thread_id, *owner_params),
                )
            await conn.commit()
        return {
            "id": mid,
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "metadata": dict(metadata or {}),
            "created_at": now,
        }

    @staticmethod
    def _message_with_metadata(row: dict) -> dict:
        """Expose parsed ``metadata`` alongside decrypted ``metadata_json``."""
        out = dict(row)
        raw = out.get("metadata_json")
        parsed: dict = {}
        if isinstance(raw, str) and raw.strip():
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    parsed = loaded
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = {}
        out["metadata"] = parsed
        return out

    async def get_messages(self, thread_id: str, user_id: str | None = None) -> list[dict]:
        """Return messages for a thread.

        When ``user_id`` is provided, requires the thread to belong to that user
        (defense-in-depth against callers that skip ``get_thread_for_user``).
        """
        async with self._conn() as conn:
            if user_id is not None:
                async with conn.execute(
                    """SELECT m.* FROM chat_messages m
                       INNER JOIN chat_threads t ON t.id = m.thread_id
                       WHERE m.thread_id = ? AND t.user_id = ?
                       ORDER BY m.created_at ASC""",
                    (thread_id, user_id),
                ) as cur:
                    rows = await cur.fetchall()
            else:
                async with conn.execute(
                    "SELECT * FROM chat_messages WHERE thread_id = ? ORDER BY created_at ASC",
                    (thread_id,),
                ) as cur:
                    rows = await cur.fetchall()
            return [
                self._message_with_metadata(self._decrypt_row("chat_messages", dict(r)))
                for r in rows
            ]
