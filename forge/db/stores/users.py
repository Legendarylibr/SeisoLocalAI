"""User account persistence."""

from __future__ import annotations

import uuid

from forge.db.stores.constants import now_iso


class UsersMixin:
    async def user_count(self) -> int:
        async with (
            self._conn() as conn,
            conn.execute("SELECT COUNT(*) AS c FROM users") as cur,
        ):
            row = await cur.fetchone()
            return int(row["c"]) if row else 0

    async def create_first_user(
        self,
        password_hash: str,
        display_name: str,
        *,
        email: str | None = None,
        nostr_pubkey: str | None = None,
    ) -> dict:
        """Atomically create the sole local user (registration is single-tenant)."""
        uid = str(uuid.uuid4())
        now = now_iso()
        normalized_name = display_name.strip()
        resolved_email = (email or f"{uid}@local.seiso").lower()
        pubkey = (nostr_pubkey or "").strip().lower() or None
        if pubkey is not None and len(pubkey) != 64:
            raise ValueError("nostr_pubkey must be 64-char hex")
        async with self._conn() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            cur = await conn.execute("SELECT COUNT(*) AS c FROM users")
            row = await cur.fetchone()
            if row and int(row["c"]) > 0:
                await conn.execute("ROLLBACK")
                raise ValueError("Registration closed — user already exists")
            await conn.execute(
                "INSERT INTO users (id, email, password_hash, display_name, nostr_pubkey, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (uid, resolved_email, password_hash, normalized_name, pubkey, now),
            )
            await conn.commit()
        return {
            "id": uid,
            "email": resolved_email,
            "display_name": normalized_name,
            "nostr_pubkey": pubkey,
            "created_at": now,
        }

    async def create_user(
        self,
        password_hash: str,
        display_name: str,
        *,
        email: str | None = None,
        nostr_pubkey: str | None = None,
    ) -> dict:
        uid = str(uuid.uuid4())
        now = now_iso()
        normalized_name = display_name.strip()
        resolved_email = (email or f"{uid}@local.seiso").lower()
        pubkey = (nostr_pubkey or "").strip().lower() or None
        if pubkey is not None and len(pubkey) != 64:
            raise ValueError("nostr_pubkey must be 64-char hex")
        async with self._conn() as conn:
            await conn.execute(
                "INSERT INTO users (id, email, password_hash, display_name, nostr_pubkey, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (uid, resolved_email, password_hash, normalized_name, pubkey, now),
            )
            await conn.commit()
        return {
            "id": uid,
            "email": resolved_email,
            "display_name": normalized_name,
            "nostr_pubkey": pubkey,
            "created_at": now,
        }

    async def get_user_by_display_name(self, display_name: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM users WHERE lower(display_name) = ?",
                (display_name.strip().lower(),),
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_sole_user(self) -> dict | None:
        """Return the single local user, if any (Forge allows one account per instance)."""
        async with (
            self._conn() as conn,
            conn.execute("SELECT * FROM users ORDER BY created_at ASC LIMIT 1") as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_user_by_email(self, email: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower(),)
            ) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_user_by_id(self, user_id: str) -> dict | None:
        async with (
            self._conn() as conn,
            conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)) as cur,
        ):
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_user_nostr_pubkey(self, user_id: str, nostr_pubkey: str) -> dict:
        """Bind account identity to a new npub (settings key rotate / import)."""
        pubkey = (nostr_pubkey or "").strip().lower()
        if len(pubkey) != 64:
            raise ValueError("nostr_pubkey must be 64-char hex")
        async with self._conn() as conn:
            cur = await conn.execute(
                "UPDATE users SET nostr_pubkey = ? WHERE id = ?",
                (pubkey, user_id),
            )
            if cur.rowcount == 0:
                raise ValueError("User not found")
            await conn.commit()
            async with conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ) as fetch:
                row = await fetch.fetchone()
        if row is None:
            raise ValueError("User not found")
        return dict(row)
