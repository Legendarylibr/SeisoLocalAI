"""Shared constants for the SQLite persistence layer."""

from __future__ import annotations

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

CREATE TABLE IF NOT EXISTS distill_rl_jobs (
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

CREATE TABLE IF NOT EXISTS hub_publish_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    result_json TEXT DEFAULT '{}',
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

CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_events (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_user ON local_models(user_id);
CREATE INDEX IF NOT EXISTS idx_models_user_created ON local_models(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_user_path ON local_models(user_id, path);
CREATE INDEX IF NOT EXISTS idx_models_user_name ON local_models(user_id, name);
CREATE INDEX IF NOT EXISTS idx_threads_user ON chat_threads(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON training_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread_created ON chat_messages(thread_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_export_jobs_user ON export_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_export_jobs_user_created ON export_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_compress_jobs_user ON compress_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_compress_jobs_user_created ON compress_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_distill_rl_jobs_user ON distill_rl_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_distill_rl_jobs_user_created ON distill_rl_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rl_quant_jobs_user ON rl_quant_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_rl_quant_jobs_user_created ON rl_quant_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hub_publish_jobs_user ON hub_publish_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_hub_publish_jobs_user_created ON hub_publish_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_providers_user ON providers(user_id);
CREATE INDEX IF NOT EXISTS idx_providers_user_created ON providers(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_events_job_sequence ON job_events(job_id, sequence ASC);
CREATE INDEX IF NOT EXISTS idx_job_events_user_created ON job_events(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_events_kind ON job_events(kind, job_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_models_user_source ON local_models(user_id, source);
CREATE INDEX IF NOT EXISTS idx_threads_user_updated ON chat_threads(user_id, updated_at DESC);
"""

_TRAINING_LIST_COLUMNS = (
    "id",
    "user_id",
    "project_id",
    "status",
    "config_json",
    "created_at",
    "updated_at",
)
_EXPORT_LIST_COLUMNS = ("id", "user_id", "status", "created_at", "updated_at")
_HUB_PUBLISH_LIST_COLUMNS = ("id", "user_id", "status", "created_at", "updated_at")
_STAGE_PIPELINE_LIST_COLUMNS = (
    "id",
    "user_id",
    "status",
    "output_dir",
    "run_dir",
    "model_dir",
    "stages_json",
    "created_at",
    "updated_at",
)
_RL_QUANT_LIST_COLUMNS = (
    "id",
    "user_id",
    "status",
    "output_dir",
    "recommendation_path",
    "gguf_quants_json",
    "created_at",
    "updated_at",
)
_UPSERT_MODEL_SQL = """INSERT INTO local_models
   (id, user_id, name, path, source, format, size_bytes, metadata_json, created_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
   ON CONFLICT(user_id, source) DO UPDATE SET
   name = excluded.name,
   path = excluded.path,
   format = excluded.format,
   size_bytes = excluded.size_bytes,
   metadata_json = excluded.metadata_json"""

_JOB_ERROR_TABLES = (
    "training_jobs",
    "export_jobs",
    "rl_quant_jobs",
    "compress_jobs",
    "distill_rl_jobs",
    "hub_publish_jobs",
)
_CONFIG_JOB_TABLES = frozenset({"rl_quant_jobs", "compress_jobs", "distill_rl_jobs"})


def column_list(columns: tuple[str, ...]) -> str:
    return ", ".join(columns)


def config_job_table(table: str) -> str:
    if table not in _CONFIG_JOB_TABLES:
        raise ValueError(f"Unsupported config job table: {table}")
    return table


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
