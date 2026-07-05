"""Small SQL helpers shared by the Forge store."""

from __future__ import annotations

from datetime import datetime, timezone

from forge.db.schema import CONFIG_JOB_TABLES


def column_list(columns: tuple[str, ...]) -> str:
    return ", ".join(columns)


def config_job_table(table: str) -> str:
    if table not in CONFIG_JOB_TABLES:
        raise ValueError(f"Unsupported config job table: {table}")
    return table


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
