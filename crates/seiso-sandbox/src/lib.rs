//! Path validation and sandboxing — port of `seiso.security`.
//!
//! Rejects traversal, unsafe characters, and symlink segments on join
//! (write-sink protection against cross-tenant escapes).

use std::path::{Component, Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SecurityError {
    #[error("Invalid path segment: {0:?}")]
    InvalidSegment(String),
    #[error("Unsafe characters in segment: {0:?}")]
    UnsafeSegment(String),
    #[error("Symlink rejected in path segment: {0:?}")]
    SymlinkRejected(String),
    #[error("Path traversal detected")]
    Traversal,
    #[error("Path {target} is outside sandbox {base}")]
    OutsideSandbox { target: String, base: String },
    #[error("Invalid user_id: {0:?}")]
    InvalidUserId(String),
    #[error("Path must be under a user-scoped root for {0}")]
    NotUserScoped(String),
    #[error("Access denied to path root: {0:?}")]
    DeniedRoot(String),
    #[error("Path must be under {root}/{user_id}/")]
    WrongOwner { root: String, user_id: String },
}

/// Characters forbidden in user-supplied path segments (`_UNSAFE_SEGMENT`).
fn has_unsafe_chars(part: &str) -> bool {
    part.chars()
        .any(|c| matches!(c, '\0' | '<' | '>' | ':' | '"' | '|' | '?' | '*'))
}

/// User-scoped data roots under `SEISO_DATA_DIR` (single source of truth).
pub const USER_SCOPED_DATA_ROOTS: &[&str] = &[
    "uploads",
    "knowledge",
    "artifacts",
    "sandbox",
    "models",
    "checkpoints",
    "exports",
    "compress",
    "distill_rl",
    "recipes",
];

fn is_user_scoped_root(root: &str) -> bool {
    USER_SCOPED_DATA_ROOTS.contains(&root)
}

fn is_within(base: &Path, target: &Path) -> bool {
    match (base.canonicalize(), target.canonicalize()) {
        (Ok(base_r), Ok(target_r)) => target_r.starts_with(&base_r),
        _ => {
            // Fall back for non-existent paths: compare components carefully
            let base_c = normalize_lex(base);
            let target_c = normalize_lex(target);
            target_c.starts_with(&base_c)
        }
    }
}

fn normalize_lex(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Join paths ensuring the result stays within `base` (no traversal).
pub fn safe_join(base: &Path, parts: &[&str]) -> Result<PathBuf, SecurityError> {
    let base = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());
    let mut candidate = base.clone();
    for part in parts {
        if part.is_empty() || *part == "." || *part == ".." {
            return Err(SecurityError::InvalidSegment((*part).to_string()));
        }
        if part.contains('/') || part.contains('\\') || has_unsafe_chars(part) {
            return Err(SecurityError::UnsafeSegment((*part).to_string()));
        }
        // Reject embedded `..` even if path lib would split (defensive)
        if part.split(['/', '\\']).any(|p| p == "..") {
            return Err(SecurityError::InvalidSegment((*part).to_string()));
        }
        let next = candidate.join(part);
        if next.is_symlink() {
            return Err(SecurityError::SymlinkRejected((*part).to_string()));
        }
        // resolve as far as possible
        let resolved = if next.exists() {
            next.canonicalize().unwrap_or(next.clone())
        } else if let Some(parent) = next.parent() {
            let parent_r = parent
                .canonicalize()
                .unwrap_or_else(|_| parent.to_path_buf());
            parent_r.join(next.file_name().unwrap_or_default())
        } else {
            next.clone()
        };
        if !is_within(&base, &resolved) && resolved.exists() {
            return Err(SecurityError::Traversal);
        }
        // Also check lexical containment for not-yet-created paths
        let lex_ok = normalize_lex(&resolved).starts_with(normalize_lex(&base));
        if !lex_ok {
            return Err(SecurityError::Traversal);
        }
        candidate = resolved;
    }
    Ok(candidate)
}

/// Verify `target` is inside `base`; return resolved target.
pub fn assert_within(base: &Path, target: &Path) -> Result<PathBuf, SecurityError> {
    let base_r = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());
    let target_r = target
        .canonicalize()
        .unwrap_or_else(|_| target.to_path_buf());
    if !is_within(&base_r, &target_r) {
        return Err(SecurityError::OutsideSandbox {
            target: target_r.display().to_string(),
            base: base_r.display().to_string(),
        });
    }
    Ok(target_r)
}

/// Require `target` under `data_dir/<scoped_root>/<user_id>/...`.
pub fn assert_user_scoped_path(
    data_dir: &Path,
    user_id: &str,
    target: &Path,
) -> Result<PathBuf, SecurityError> {
    if user_id.is_empty()
        || user_id.contains('/')
        || user_id.contains('\\')
        || user_id == "."
        || user_id == ".."
    {
        return Err(SecurityError::InvalidUserId(user_id.to_string()));
    }
    let base = data_dir
        .canonicalize()
        .unwrap_or_else(|_| data_dir.to_path_buf());
    let target_r = assert_within(&base, target)?;
    let rel = target_r
        .strip_prefix(&base)
        .map_err(|_| SecurityError::OutsideSandbox {
            target: target_r.display().to_string(),
            base: base.display().to_string(),
        })?;
    let mut parts = rel.components();
    let root = parts
        .next()
        .and_then(|c| c.as_os_str().to_str())
        .ok_or_else(|| SecurityError::NotUserScoped(user_id.to_string()))?;
    let owner = parts
        .next()
        .and_then(|c| c.as_os_str().to_str())
        .ok_or_else(|| SecurityError::NotUserScoped(user_id.to_string()))?;
    if !is_user_scoped_root(root) {
        return Err(SecurityError::DeniedRoot(root.to_string()));
    }
    if owner != user_id {
        return Err(SecurityError::WrongOwner {
            root: root.to_string(),
            user_id: user_id.to_string(),
        });
    }
    Ok(target_r)
}

/// Produce a safe filename from user input.
pub fn sanitize_filename(name: &str, max_len: usize) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let cleaned = cleaned.trim_matches(|c| c == '.' || c == ' ');
    let cleaned = if cleaned.is_empty() {
        "unnamed"
    } else {
        cleaned
    };
    cleaned.chars().take(max_len).collect()
}

/// Reject empty, absolute, or `..` relative artifact names.
pub fn assert_relative_artifact_name(name: &str, field: &str) -> Result<String, String> {
    let raw = name.trim();
    if raw.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    let path = Path::new(raw);
    if path.is_absolute() {
        return Err(format!("{field} must be a relative path, got {raw:?}"));
    }
    for c in path.components() {
        match c {
            Component::ParentDir => {
                return Err(format!("{field} must not contain '..' path segments"));
            }
            Component::CurDir => {
                return Err(format!("{field} has empty or '.' path segments"));
            }
            Component::Normal(s) if s.is_empty() => {
                return Err(format!("{field} has empty or '.' path segments"));
            }
            _ => {}
        }
    }
    Ok(raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn safe_join_ok() {
        let tmp = std::env::temp_dir().join(format!("seiso-sandbox-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let p = safe_join(&tmp, &["models", "alice"]).unwrap();
        assert!(p.ends_with("models/alice") || p.ends_with(r"models\alice"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn safe_join_rejects_dotdot() {
        let tmp = std::env::temp_dir().join(format!("seiso-sandbox-dd-{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();
        assert!(matches!(
            safe_join(&tmp, &[".."]),
            Err(SecurityError::InvalidSegment(_))
        ));
        assert!(matches!(
            safe_join(&tmp, &["foo/bar"]),
            Err(SecurityError::UnsafeSegment(_))
        ));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn relative_artifact_rejects_escape() {
        assert!(assert_relative_artifact_name("../x", "path").is_err());
        assert!(assert_relative_artifact_name("/abs", "path").is_err());
        assert_eq!(
            assert_relative_artifact_name("checkpoint-best", "path").unwrap(),
            "checkpoint-best"
        );
    }

    #[test]
    fn sanitize_filename_basic() {
        assert_eq!(sanitize_filename("a/b:c", 255), "a_b_c");
    }
}
