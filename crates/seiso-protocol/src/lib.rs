//! Versioned JSONL worker protocol (control plane ↔ Python ML workers).
//!
//! One message per line. Current version: **1**.
//!
//! ```json
//! {"v":1,"op":"train.start","job_id":"...","config":{},"paths":{}}
//! {"v":1,"op":"log","job_id":"...","level":"info","msg":"..."}
//! {"v":1,"op":"done","job_id":"...","status":"ok","artifacts":[]}
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Protocol version carried on every message.
pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported protocol version {0} (need {PROTOCOL_VERSION})")]
    Version(u32),
    #[error("unknown op: {0}")]
    UnknownOp(String),
}

/// Worker / supervisor message envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub v: u32,
    pub op: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paths: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub msg: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pct: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl Message {
    pub fn new_op(op: impl Into<String>) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            op: op.into(),
            job_id: None,
            config: None,
            paths: None,
            level: None,
            msg: None,
            name: None,
            value: None,
            step: None,
            pct: None,
            status: None,
            artifacts: None,
            code: None,
        }
    }

    pub fn train_start(job_id: impl Into<String>, config: Value, paths: Value) -> Self {
        let mut m = Self::new_op("train.start");
        m.job_id = Some(job_id.into());
        m.config = Some(config);
        m.paths = Some(paths);
        m
    }

    pub fn log(
        job_id: impl Into<String>,
        level: impl Into<String>,
        msg: impl Into<String>,
    ) -> Self {
        let mut m = Self::new_op("log");
        m.job_id = Some(job_id.into());
        m.level = Some(level.into());
        m.msg = Some(msg.into());
        m
    }

    pub fn metric(
        job_id: impl Into<String>,
        name: impl Into<String>,
        value: f64,
        step: Option<u64>,
    ) -> Self {
        let mut m = Self::new_op("metric");
        m.job_id = Some(job_id.into());
        m.name = Some(name.into());
        m.value = Some(value);
        m.step = step;
        m
    }

    pub fn progress(job_id: impl Into<String>, pct: f64) -> Self {
        let mut m = Self::new_op("progress");
        m.job_id = Some(job_id.into());
        m.pct = Some(pct);
        m
    }

    pub fn done(
        job_id: impl Into<String>,
        status: impl Into<String>,
        artifacts: Vec<String>,
    ) -> Self {
        let mut m = Self::new_op("done");
        m.job_id = Some(job_id.into());
        m.status = Some(status.into());
        m.artifacts = Some(artifacts);
        m
    }

    pub fn error(
        job_id: impl Into<String>,
        code: impl Into<String>,
        msg: impl Into<String>,
    ) -> Self {
        let mut m = Self::new_op("error");
        m.job_id = Some(job_id.into());
        m.code = Some(code.into());
        m.msg = Some(msg.into());
        m
    }

    pub fn cancel(job_id: impl Into<String>) -> Self {
        let mut m = Self::new_op("cancel");
        m.job_id = Some(job_id.into());
        m
    }

    pub fn validate_version(&self) -> Result<(), ProtocolError> {
        if self.v != PROTOCOL_VERSION {
            return Err(ProtocolError::Version(self.v));
        }
        Ok(())
    }

    pub fn to_jsonl(&self) -> Result<String, ProtocolError> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_jsonl(line: &str) -> Result<Self, ProtocolError> {
        let msg: Self = serde_json::from_str(line.trim())?;
        msg.validate_version()?;
        Ok(msg)
    }
}

/// Well-known ops.
pub mod ops {
    pub const TRAIN_START: &str = "train.start";
    pub const EXPORT_START: &str = "export.start";
    pub const COMPRESS_START: &str = "compress.start";
    pub const DISTILL_RL_START: &str = "distill_rl.start";
    pub const LOG: &str = "log";
    pub const METRIC: &str = "metric";
    pub const PROGRESS: &str = "progress";
    pub const DONE: &str = "done";
    pub const ERROR: &str = "error";
    pub const CANCEL: &str = "cancel";
    pub const HEARTBEAT: &str = "heartbeat";
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_train_start() {
        let m = Message::train_start("j1", json!({"lr": 1e-4}), json!({"data_dir": "/tmp"}));
        let line = m.to_jsonl().unwrap();
        let back = Message::from_jsonl(&line).unwrap();
        assert_eq!(back.op, ops::TRAIN_START);
        assert_eq!(back.job_id.as_deref(), Some("j1"));
        assert_eq!(back.v, PROTOCOL_VERSION);
    }

    #[test]
    fn reject_bad_version() {
        let line = r#"{"v":99,"op":"log","job_id":"x","level":"info","msg":"hi"}"#;
        assert!(Message::from_jsonl(line).is_err());
    }

    #[test]
    fn log_line() {
        let m = Message::log("j", "info", "hello");
        let parsed = Message::from_jsonl(&m.to_jsonl().unwrap()).unwrap();
        assert_eq!(parsed.msg.as_deref(), Some("hello"));
    }
}
