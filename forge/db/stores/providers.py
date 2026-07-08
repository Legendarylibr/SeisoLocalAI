"""Remote provider configuration persistence."""

from __future__ import annotations

import json
import uuid

from forge.db.stores.constants import now_iso


class ProvidersMixin:
    async def list_providers(self, user_id: str) -> list[dict]:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM providers WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ) as cur,
        ):
            return [
                self._decrypt_row("providers", dict(r)) for r in await cur.fetchall()
            ]

    async def create_provider(
        self, user_id: str, name: str, provider_type: str, config: dict
    ) -> dict:
        pid = str(uuid.uuid4())
        now = now_iso()
        enc_config = self._enc(json.dumps(config))
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO providers (id, user_id, name, provider_type, config_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pid, user_id, name, provider_type, enc_config, now),
            )
            await conn.commit()
        return {
            "id": pid,
            "name": name,
            "provider_type": provider_type,
            "config": config,
            "created_at": now,
        }

    async def get_provider(self, provider_id: str, user_id: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM providers WHERE id = ? AND user_id = ?",
                (provider_id, user_id),
            ) as cur,
        ):
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
