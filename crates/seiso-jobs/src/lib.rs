//! In-process job registry and Python worker supervisor.
//!
//! Spawns `python -m seiso_ml_worker` (or a configured command) and speaks
//! [`seiso_protocol`] JSONL over stdin/stdout.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use seiso_protocol::{ops, Message};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{broadcast, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum JobError {
    #[error("job not found: {0}")]
    NotFound(String),
    #[error("job already terminal: {0}")]
    Terminal(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("protocol: {0}")]
    Protocol(#[from] seiso_protocol::ProtocolError),
    #[error("worker failed to start: {0}")]
    Start(String),
    #[error("unsupported job kind: {0}")]
    UnsupportedKind(String),
    #[error("job timed out: {0}")]
    Timeout(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub kind: String,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub config: Value,
    pub logs: Vec<String>,
    pub error: Option<String>,
    pub artifacts: Vec<String>,
}

impl JobRecord {
    pub fn new(kind: impl Into<String>, config: Value) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            kind: kind.into(),
            status: JobStatus::Queued,
            created_at: now,
            updated_at: now,
            config,
            logs: Vec::new(),
            error: None,
            artifacts: Vec::new(),
        }
    }
}

/// Events streamed to SSE subscribers.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    Log {
        job_id: String,
        message: String,
    },
    Metric {
        job_id: String,
        name: String,
        value: f64,
        step: Option<u64>,
    },
    Progress {
        job_id: String,
        pct: f64,
    },
    Status {
        job_id: String,
        status: JobStatus,
    },
    Done {
        job_id: String,
        status: JobStatus,
        artifacts: Vec<String>,
    },
    Error {
        job_id: String,
        message: String,
    },
}

/// How to locate and run the Python ML worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub python: PathBuf,
    pub module: String,
    /// Directories prepended to `PYTHONPATH` (so `seiso_ml_worker` resolves).
    pub pythonpath: Vec<PathBuf>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        let mut pythonpath = Vec::new();
        // Repo-layout defaults when running from checkout.
        for candidate in ["python", "../python", "../../python"] {
            let p = PathBuf::from(candidate);
            if p.is_dir() {
                pythonpath.push(p);
            }
        }
        if let Ok(extra) = std::env::var("SEISO_ML_WORKER_PYTHONPATH") {
            for part in extra.split(':').filter(|s| !s.is_empty()) {
                pythonpath.push(PathBuf::from(part));
            }
        }
        Self {
            python: PathBuf::from(
                std::env::var("SEISO_PYTHON").unwrap_or_else(|_| "python3".into()),
            ),
            module: std::env::var("SEISO_ML_WORKER_MODULE")
                .unwrap_or_else(|_| "seiso_ml_worker".into()),
            pythonpath,
        }
    }
}

#[derive(Clone)]
pub struct JobSupervisor {
    inner: Arc<RwLock<Inner>>,
    events: broadcast::Sender<JobEvent>,
    worker: WorkerConfig,
    data_dir: PathBuf,
}

struct Inner {
    jobs: HashMap<String, JobRecord>,
    children: HashMap<String, Child>,
}

impl JobSupervisor {
    pub fn new(data_dir: PathBuf) -> Self {
        Self::with_worker(data_dir, WorkerConfig::default())
    }

    pub fn with_worker(data_dir: PathBuf, worker: WorkerConfig) -> Self {
        let (events, _) = broadcast::channel(512);
        Self {
            inner: Arc::new(RwLock::new(Inner {
                jobs: HashMap::new(),
                children: HashMap::new(),
            })),
            events,
            worker,
            data_dir,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<JobEvent> {
        self.events.subscribe()
    }

    pub async fn get(&self, id: &str) -> Option<JobRecord> {
        self.inner.read().await.jobs.get(id).cloned()
    }

    pub async fn list(&self) -> Vec<JobRecord> {
        let mut jobs: Vec<_> = self.inner.read().await.jobs.values().cloned().collect();
        jobs.sort_by_key(|b| std::cmp::Reverse(b.created_at));
        jobs
    }

    /// Start a job by kind (`train` | `export`).
    pub async fn start_job(&self, kind: &str, config: Value) -> Result<JobRecord, JobError> {
        let op = match kind {
            "train" => ops::TRAIN_START,
            "export" => ops::EXPORT_START,
            other => return Err(JobError::UnsupportedKind(other.to_string())),
        };

        let mut job = JobRecord::new(kind, config.clone());
        let job_id = job.id.clone();
        job.status = JobStatus::Running;
        job.updated_at = Utc::now();

        {
            let mut g = self.inner.write().await;
            g.jobs.insert(job_id.clone(), job.clone());
        }
        let _ = self.events.send(JobEvent::Status {
            job_id: job_id.clone(),
            status: JobStatus::Running,
        });

        let paths = serde_json::json!({
            "data_dir": self.data_dir.display().to_string(),
        });
        let mut start_msg = Message::new_op(op);
        start_msg.job_id = Some(job_id.clone());
        start_msg.config = Some(config);
        start_msg.paths = Some(paths);

        let mut cmd = Command::new(&self.worker.python);
        cmd.arg("-m")
            .arg(&self.worker.module)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("SEISO_DATA_DIR", &self.data_dir)
            .env("PYTHONUNBUFFERED", "1");

        if !self.worker.pythonpath.is_empty() {
            let joined = self
                .worker
                .pythonpath
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(":");
            let existing = std::env::var("PYTHONPATH").unwrap_or_default();
            let path = if existing.is_empty() {
                joined
            } else {
                format!("{joined}:{existing}")
            };
            cmd.env("PYTHONPATH", path);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| JobError::Start(format!("{} ({e})", self.worker.python.display())))?;

        if let Some(mut stdin) = child.stdin.take() {
            let line = start_msg.to_jsonl()?;
            stdin.write_all(line.as_bytes()).await?;
            stdin.write_all(b"\n").await?;
            stdin.flush().await?;
            drop(stdin);
        }

        {
            let mut g = self.inner.write().await;
            g.children.insert(job_id.clone(), child);
        }

        let this = self.clone();
        let jid = job_id.clone();
        tokio::spawn(async move {
            if let Err(e) = this.pump_worker(&jid).await {
                warn!(job_id = %jid, error = %e, "worker pump ended with error");
            }
        });

        Ok(job)
    }

    pub async fn start_train(&self, config: Value) -> Result<JobRecord, JobError> {
        self.start_job("train", config).await
    }

    pub async fn start_export(&self, config: Value) -> Result<JobRecord, JobError> {
        self.start_job("export", config).await
    }

    /// Poll until the job reaches a terminal status or `timeout` elapses.
    pub async fn wait_until_terminal(
        &self,
        job_id: &str,
        timeout: Duration,
    ) -> Result<JobRecord, JobError> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if let Some(job) = self.get(job_id).await {
                if job.status.is_terminal() {
                    return Ok(job);
                }
            } else {
                return Err(JobError::NotFound(job_id.to_string()));
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(JobError::Timeout(job_id.to_string()));
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    async fn pump_worker(&self, job_id: &str) -> Result<(), JobError> {
        let mut stdout = {
            let mut g = self.inner.write().await;
            let child = g
                .children
                .get_mut(job_id)
                .ok_or_else(|| JobError::NotFound(job_id.to_string()))?;
            child
                .stdout
                .take()
                .ok_or_else(|| JobError::Start("missing stdout".into()))?
        };

        let mut reader = BufReader::new(&mut stdout).lines();
        while let Some(line) = reader.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            match Message::from_jsonl(&line) {
                Ok(msg) => self.handle_worker_msg(job_id, msg).await?,
                Err(e) => {
                    warn!(%line, error = %e, "bad worker line");
                    self.append_log(job_id, &format!("[worker raw] {line}"))
                        .await;
                }
            }
        }

        // Drain a bit of stderr into logs for debugging failed starts
        {
            let mut g = self.inner.write().await;
            if let Some(child) = g.children.get_mut(job_id) {
                if let Some(mut err) = child.stderr.take() {
                    drop(g);
                    let mut err_reader = BufReader::new(&mut err).lines();
                    while let Ok(Some(line)) = err_reader.next_line().await {
                        if !line.trim().is_empty() {
                            self.append_log(job_id, &format!("[stderr] {line}")).await;
                        }
                    }
                }
            }
        }

        let status = {
            let mut g = self.inner.write().await;
            if let Some(mut child) = g.children.remove(job_id) {
                child.wait().await?
            } else {
                return Ok(());
            }
        };

        let mut g = self.inner.write().await;
        if let Some(job) = g.jobs.get_mut(job_id) {
            if !job.status.is_terminal() {
                if status.success() {
                    job.status = JobStatus::Succeeded;
                } else {
                    job.status = JobStatus::Failed;
                    job.error = Some(format!("worker exit {status}"));
                }
                job.updated_at = Utc::now();
                let st = job.status;
                let arts = job.artifacts.clone();
                drop(g);
                let _ = self.events.send(JobEvent::Done {
                    job_id: job_id.to_string(),
                    status: st,
                    artifacts: arts,
                });
            }
        }
        Ok(())
    }

    async fn handle_worker_msg(&self, job_id: &str, msg: Message) -> Result<(), JobError> {
        match msg.op.as_str() {
            ops::LOG => {
                let text = msg.msg.unwrap_or_default();
                self.append_log(job_id, &text).await;
                let _ = self.events.send(JobEvent::Log {
                    job_id: job_id.to_string(),
                    message: text,
                });
            }
            ops::METRIC => {
                if let (Some(name), Some(value)) = (msg.name, msg.value) {
                    let _ = self.events.send(JobEvent::Metric {
                        job_id: job_id.to_string(),
                        name,
                        value,
                        step: msg.step,
                    });
                }
            }
            ops::PROGRESS => {
                if let Some(pct) = msg.pct {
                    let _ = self.events.send(JobEvent::Progress {
                        job_id: job_id.to_string(),
                        pct,
                    });
                }
            }
            ops::DONE => {
                let mut g = self.inner.write().await;
                if let Some(job) = g.jobs.get_mut(job_id) {
                    job.status = match msg.status.as_deref() {
                        Some("ok") => JobStatus::Succeeded,
                        Some("cancelled") => JobStatus::Cancelled,
                        _ => JobStatus::Failed,
                    };
                    job.artifacts = msg.artifacts.unwrap_or_default();
                    job.updated_at = Utc::now();
                    let st = job.status;
                    let arts = job.artifacts.clone();
                    drop(g);
                    let _ = self.events.send(JobEvent::Done {
                        job_id: job_id.to_string(),
                        status: st,
                        artifacts: arts,
                    });
                }
            }
            ops::ERROR => {
                let err = msg.msg.unwrap_or_else(|| "worker error".into());
                let mut g = self.inner.write().await;
                if let Some(job) = g.jobs.get_mut(job_id) {
                    job.status = JobStatus::Failed;
                    job.error = Some(err.clone());
                    job.updated_at = Utc::now();
                }
                drop(g);
                let _ = self.events.send(JobEvent::Error {
                    job_id: job_id.to_string(),
                    message: err,
                });
            }
            ops::HEARTBEAT => {
                info!(job_id, "worker heartbeat");
            }
            other => {
                warn!(op = other, "unhandled worker op");
            }
        }
        Ok(())
    }

    async fn append_log(&self, job_id: &str, line: &str) {
        let mut g = self.inner.write().await;
        if let Some(job) = g.jobs.get_mut(job_id) {
            job.logs.push(line.to_string());
            if job.logs.len() > 5000 {
                let drain = job.logs.len() - 5000;
                job.logs.drain(0..drain);
            }
            job.updated_at = Utc::now();
        }
    }

    pub async fn cancel(&self, job_id: &str) -> Result<JobRecord, JobError> {
        let mut g = self.inner.write().await;
        let job = g
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| JobError::NotFound(job_id.to_string()))?;
        if job.status.is_terminal() {
            return Err(JobError::Terminal(job_id.to_string()));
        }
        job.status = JobStatus::Cancelled;
        job.updated_at = Utc::now();
        let record = job.clone();
        if let Some(mut child) = g.children.remove(job_id) {
            let _ = child.kill().await;
        }
        drop(g);
        let _ = self.events.send(JobEvent::Status {
            job_id: job_id.to_string(),
            status: JobStatus::Cancelled,
        });
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn repo_python_path() -> PathBuf {
        // crates/seiso-jobs -> repo/python
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        here.join("../../python").canonicalize().unwrap()
    }

    #[test]
    fn job_record_defaults() {
        let j = JobRecord::new("train", serde_json::json!({}));
        assert_eq!(j.status, JobStatus::Queued);
        assert_eq!(j.kind, "train");
        assert!(!j.id.is_empty());
    }

    #[tokio::test]
    async fn e2e_train_smoke_via_worker() {
        let tmp = std::env::temp_dir().join(format!("seiso-jobs-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).unwrap();
        let worker = WorkerConfig {
            python: PathBuf::from("python3"),
            module: "seiso_ml_worker".into(),
            pythonpath: vec![repo_python_path()],
        };
        let sup = JobSupervisor::with_worker(tmp.clone(), worker);
        let job = sup
            .start_train(serde_json::json!({"smoke_only": true}))
            .await
            .expect("start train");
        let done = sup
            .wait_until_terminal(&job.id, Duration::from_secs(15))
            .await
            .expect("wait");
        assert_eq!(done.status, JobStatus::Succeeded, "logs={:?}", done.logs);
        assert!(
            done.logs.iter().any(|l| l.contains("smoke train complete")),
            "logs={:?}",
            done.logs
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn e2e_export_smoke_via_worker() {
        let tmp = std::env::temp_dir().join(format!("seiso-jobs-exp-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).unwrap();
        let worker = WorkerConfig {
            python: PathBuf::from("python3"),
            module: "seiso_ml_worker".into(),
            pythonpath: vec![repo_python_path()],
        };
        let sup = JobSupervisor::with_worker(tmp.clone(), worker);
        let job = sup
            .start_export(serde_json::json!({"smoke_only": true, "format": "gguf"}))
            .await
            .expect("start export");
        let done = sup
            .wait_until_terminal(&job.id, Duration::from_secs(15))
            .await
            .expect("wait");
        assert_eq!(done.status, JobStatus::Succeeded, "logs={:?}", done.logs);
        assert!(
            done.artifacts.iter().any(|a| a.contains("export")),
            "artifacts={:?}",
            done.artifacts
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
