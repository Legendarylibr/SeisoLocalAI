//! SQLite store for the Rust control plane.
//!
//! Schema is intentionally minimal in Phase 1 and will grow toward parity
//! with Python `forge/db`. Encrypted fields use `seiso-crypto` (`enc:v1:`).

use std::path::Path;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("sqlx: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("crypto: {0}")]
    Crypto(#[from] seiso_crypto::CryptoError),
    #[error("{0}")]
    Other(String),
}

#[derive(Clone)]
pub struct Db {
    pool: SqlitePool,
}

impl Db {
    pub async fn connect(path: &Path) -> Result<Self, DbError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DbError::Other(format!("create data dir: {e}")))?;
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;
        let db = Self { pool };
        db.migrate().await?;
        info!(path = %path.display(), "sqlite ready");
        Ok(db)
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    async fn migrate(&self) -> Result<(), DbError> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY NOT NULL,
                npub TEXT UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                settings_json TEXT
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Schema version
        sqlx::query(
            r#"
            INSERT INTO meta(key, value) VALUES('schema_version', '1')
            ON CONFLICT(key) DO UPDATE SET value=excluded.value;
            "#,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn schema_version(&self) -> Result<String, DbError> {
        let row: (String,) = sqlx::query_as("SELECT value FROM meta WHERE key = 'schema_version'")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.0)
    }

    /// Store an encrypted setting (value wrapped with enc:v1 when key provided).
    pub async fn put_encrypted_setting(
        &self,
        user_id: &str,
        key_name: &str,
        plaintext: &str,
        encryption_key: &[u8; 32],
    ) -> Result<(), DbError> {
        let enc = seiso_crypto::encrypt_field(plaintext, encryption_key)?;
        // Merge into settings_json naively
        let existing: Option<(Option<String>,)> =
            sqlx::query_as("SELECT settings_json FROM users WHERE id = ?")
                .bind(user_id)
                .fetch_optional(&self.pool)
                .await?;
        let mut map: serde_json::Map<String, serde_json::Value> = match existing {
            Some((Some(raw),)) => serde_json::from_str(&raw).unwrap_or_default(),
            _ => serde_json::Map::new(),
        };
        map.insert(key_name.to_string(), serde_json::Value::String(enc));
        let json = serde_json::Value::Object(map).to_string();
        sqlx::query(
            r#"
            INSERT INTO users(id, settings_json) VALUES(?, ?)
            ON CONFLICT(id) DO UPDATE SET settings_json=excluded.settings_json
            "#,
        )
        .bind(user_id)
        .bind(json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
