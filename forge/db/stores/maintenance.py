"""Cross-table maintenance operations."""

from __future__ import annotations

from forge.db.stores.constants import _JOB_ERROR_TABLES, now_iso


class MaintenanceMixin:
    async def reset_local_session(self) -> dict[str, int]:
        """Clear local users and user-owned workspace rows for forgotten-password reset."""
        async with self._conn() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            counts: dict[str, int] = {}
            for table in (
                "chat_messages",
                "chat_threads",
                "providers",
                "knowledge_bases",
                "recipe_jobs",
                "distill_rl_jobs",
                "compress_jobs",
                "rl_quant_jobs",
                "export_jobs",
                "training_jobs",
                "local_models",
                "projects",
                "users",
            ):
                cur = await conn.execute(f"SELECT COUNT(*) AS c FROM {table}")
                row = await cur.fetchone()
                counts[table] = int(row["c"]) if row else 0
                await conn.execute(f"DELETE FROM {table}")
            await conn.commit()
        return counts

    async def reconcile_stale_jobs(
        self, *, reason: str = "Server restarted while job was active"
    ) -> int:
        """Mark in-flight jobs as failed after Forge restart (orchestrator state is in-memory only)."""
        now = now_iso()
        total = 0
        async with self._conn() as conn:
            for table in _JOB_ERROR_TABLES:
                query = f"""UPDATE {table}
                        SET status = 'failed', updated_at = ?,
                            error_text = COALESCE(error_text, ?)
                        WHERE status IN ('pending', 'running')"""  # nosec B608
                cur = await conn.execute(
                    query,
                    (now, reason),
                )
                total += cur.rowcount
            await conn.commit()
        return total
