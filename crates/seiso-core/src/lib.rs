//! Shared configuration, paths, and environment helpers.
//!
//! Mirrors the subset of `seiso/env.py` and `forge/config.py` needed by the
//! Rust control plane. Python Forge remains the reference implementation until
//! cutover (`SEISO_FORGE_IMPL`).

use std::env;
use std::path::{Path, PathBuf};

use thiserror::Error;

/// Default Forge listen port (matches Python `ForgeSettings.port`).
pub const DEFAULT_PORT: u16 = 8765;

/// Default bind host — localhost only.
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// Env switch for dual-run: `python` (default) or `rust`.
pub const FORGE_IMPL_ENV: &str = "SEISO_FORGE_IMPL";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgeImpl {
    Python,
    Rust,
}

impl ForgeImpl {
    pub fn from_env() -> Self {
        match env_str(FORGE_IMPL_ENV, "python")
            .to_ascii_lowercase()
            .as_str()
        {
            "rust" | "rs" => Self::Rust,
            _ => Self::Python,
        }
    }
}

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Parse a boolean env var (`1/true/yes/on`).
pub fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(raw) => {
            let v = raw.trim().to_ascii_lowercase();
            if v.is_empty() {
                return default;
            }
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => default,
    }
}

/// Parse an integer env var, falling back to `default` on missing/invalid.
pub fn env_int(name: &str, default: i64) -> i64 {
    match env::var(name) {
        Ok(raw) => {
            let v = raw.trim();
            if v.is_empty() {
                return default;
            }
            v.parse().unwrap_or(default)
        }
        Err(_) => default,
    }
}

/// Parse a string env var with default when unset or blank.
pub fn env_str(name: &str, default: &str) -> String {
    match env::var(name) {
        Ok(raw) => {
            let v = raw.trim();
            if v.is_empty() {
                default.to_string()
            } else {
                v.to_string()
            }
        }
        Err(_) => default.to_string(),
    }
}

/// Expand `~` and resolve the Seiso data directory (`SEISO_DATA_DIR`, default `~/.seiso`).
pub fn resolve_data_dir(raw: Option<&str>) -> Result<PathBuf, CoreError> {
    let raw = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| env_str("SEISO_DATA_DIR", "~/.seiso"));
    let path = expand_user(&raw);
    std::fs::create_dir_all(&path)?;
    let path = path.canonicalize().unwrap_or(path);
    Ok(path)
}

/// Expand a leading `~` or `~/` using `$HOME` / `$USERPROFILE`.
pub fn expand_user(raw: &str) -> PathBuf {
    if raw == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from("."));
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(raw)
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Runtime settings for the Rust Forge binary.
#[derive(Debug, Clone)]
pub struct ForgeSettings {
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    /// When true, bind only loopback (always true in v1 scaffold).
    pub localhost_only: bool,
    /// Optional path to built forge-ui dist for SPA serving.
    pub ui_dist: Option<PathBuf>,
}

impl ForgeSettings {
    pub fn from_env() -> Result<Self, CoreError> {
        let host = env_str("SEISO_HOST", DEFAULT_HOST);
        let port = env_int("SEISO_PORT", DEFAULT_PORT as i64) as u16;
        if port == 0 {
            return Err(CoreError::Config("SEISO_PORT must be non-zero".into()));
        }
        let data_dir = resolve_data_dir(None)?;
        let ui_dist = env::var("SEISO_UI_DIST")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(default_ui_dist);
        Ok(Self {
            host,
            port,
            data_dir,
            localhost_only: true,
            ui_dist,
        })
    }

    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn forge_db_path(&self) -> PathBuf {
        self.data_dir.join("forge.db")
    }
}

fn default_ui_dist() -> Option<PathBuf> {
    // Prefer repo-relative forge-ui/dist when running from a checkout.
    let candidates = [
        Path::new("forge-ui/dist"),
        Path::new("../forge-ui/dist"),
        Path::new("../../forge-ui/dist"),
    ];
    for c in candidates {
        if c.is_dir() {
            return Some(c.to_path_buf());
        }
    }
    None
}

/// Subdirectories under `SEISO_DATA_DIR` (product layout).
pub mod layout {
    pub const HF_CACHE: &str = "hf_cache";
    pub const MODELS: &str = "models";
    pub const CHECKPOINTS: &str = "checkpoints";
    pub const EXPORTS: &str = "exports";
    pub const COMPRESS: &str = "compress";
    pub const DISTILL_RL: &str = "distill_rl";
    pub const KNOWLEDGE: &str = "knowledge";
    pub const NOSTR_KEYS: &str = "nostr_keys";
    pub const UPLOADS: &str = "uploads";
    pub const ARTIFACTS: &str = "artifacts";
    pub const SANDBOX: &str = "sandbox";
    pub const RECIPES: &str = "recipes";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_bool_truthy() {
        env::set_var("SEISO_TEST_BOOL", "YES");
        assert!(env_bool("SEISO_TEST_BOOL", false));
        env::remove_var("SEISO_TEST_BOOL");
    }

    #[test]
    fn expand_user_home() {
        if home_dir().is_some() {
            let p = expand_user("~/Seiso");
            assert!(p.is_absolute() || p.starts_with(home_dir().unwrap()));
        }
    }

    #[test]
    fn forge_impl_default_python() {
        env::remove_var(FORGE_IMPL_ENV);
        assert_eq!(ForgeImpl::from_env(), ForgeImpl::Python);
    }
}
