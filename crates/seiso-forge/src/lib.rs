//! Seiso Forge HTTP control plane (library surface for tests + binary).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use seiso_core::ForgeSettings;
use seiso_crypto::{decrypt_field, encrypt_field, generate_encryption_key, resolve_encryption_key};
use seiso_db::Db;
use seiso_jobs::{JobEvent, JobRecord, JobSupervisor, WorkerConfig};
use seiso_sandbox::{assert_relative_artifact_name, safe_join};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast::error::RecvError;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::info;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<ForgeSettings>,
    pub db: Db,
    pub jobs: JobSupervisor,
}

/// Build the axum router (API + optional SPA).
pub fn build_router(state: AppState) -> Router {
    let api = Router::new()
        .route("/api/health", get(health))
        .route("/api/system", get(system))
        .route("/api/jobs", get(list_jobs).post(create_job))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/jobs/{id}/cancel", post(cancel_job))
        .route("/api/jobs/{id}/events", get(job_events_sse))
        .route("/api/crypto/roundtrip", post(crypto_roundtrip))
        .route("/api/sandbox/join", post(sandbox_join))
        .with_state(state.clone());

    if let Some(dist) = state.settings.ui_dist.as_ref() {
        info!(path = %dist.display(), "serving forge-ui dist");
        let index = dist.join("index.html");
        Router::new()
            .merge(api)
            .fallback_service(ServeDir::new(dist).not_found_service(ServeFile::new(index)))
            .layer(TraceLayer::new_for_http())
    } else {
        Router::new()
            .merge(api)
            .fallback(get(|| async {
                (
                    StatusCode::NOT_FOUND,
                    Json(json!({
                        "error": "UI not found; set SEISO_UI_DIST or build forge-ui/dist",
                        "hint": "API is up at /api/health"
                    })),
                )
            }))
            .layer(TraceLayer::new_for_http())
    }
}

/// Construct app state from settings (opens SQLite, configures worker PYTHONPATH).
pub async fn app_state_from_settings(settings: ForgeSettings) -> Result<AppState, anyhow::Error> {
    let db = Db::connect(&settings.forge_db_path()).await?;
    let mut worker = WorkerConfig::default();
    // Prefer repo python/ when running from a checkout.
    let repo_python = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python");
    if let Ok(canon) = repo_python.canonicalize() {
        if !worker.pythonpath.iter().any(|p| p == &canon) {
            worker.pythonpath.insert(0, canon);
        }
    }
    let jobs = JobSupervisor::with_worker(settings.data_dir.clone(), worker);
    Ok(AppState {
        settings: Arc::new(settings),
        db,
        jobs,
    })
}

/// Bind and serve forever (used by the binary).
pub async fn serve(settings: ForgeSettings) -> anyhow::Result<()> {
    let state = app_state_from_settings(settings.clone()).await?;
    let app = build_router(state);
    let addr: SocketAddr = settings.bind_addr().parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(%addr, "listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            info!("shutdown signal");
        })
        .await?;
    Ok(())
}

async fn health(State(st): State<AppState>) -> Result<Json<Value>, AppError> {
    let schema = st.db.schema_version().await.map_err(AppError::from)?;
    Ok(Json(json!({
        "status": "ok",
        "impl": "rust",
        "version": env!("CARGO_PKG_VERSION"),
        "schema_version": schema,
        "data_dir": st.settings.data_dir.display().to_string(),
    })))
}

async fn system(State(st): State<AppState>) -> Json<Value> {
    Json(json!({
        "host": st.settings.host,
        "port": st.settings.port,
        "data_dir": st.settings.data_dir.display().to_string(),
        "forge_impl": "rust",
        "features": {
            "jobs": true,
            "jobs_export": true,
            "sse": true,
            "crypto_roundtrip": true,
            "sandbox": true,
            "auth": false,
            "inference": false,
            "phase": 1
        }
    }))
}

#[derive(Deserialize)]
struct CreateJobBody {
    #[serde(default = "default_kind")]
    kind: String,
    #[serde(default)]
    config: Value,
}

fn default_kind() -> String {
    "train".into()
}

async fn create_job(
    State(st): State<AppState>,
    Json(body): Json<CreateJobBody>,
) -> Result<(StatusCode, Json<JobRecord>), AppError> {
    let mut config = body.config;
    if config.is_null() {
        config = json!({});
    }
    // Default smoke mode for E2E / early phase.
    if config.get("smoke_only").is_none() {
        if let Some(obj) = config.as_object_mut() {
            obj.insert("smoke_only".into(), Value::Bool(true));
        }
    }
    let job = st
        .jobs
        .start_job(&body.kind, config)
        .await
        .map_err(AppError::from)?;
    Ok((StatusCode::ACCEPTED, Json(job)))
}

async fn list_jobs(State(st): State<AppState>) -> Json<Value> {
    let jobs = st.jobs.list().await;
    Json(json!({ "jobs": jobs }))
}

async fn get_job(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<JobRecord>, AppError> {
    st.jobs
        .get(&id)
        .await
        .map(Json)
        .ok_or_else(|| AppError::not_found(format!("job {id}")))
}

async fn cancel_job(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<JobRecord>, AppError> {
    st.jobs.cancel(&id).await.map(Json).map_err(AppError::from)
}

async fn job_events_sse(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>, AppError>
{
    // Ensure job exists
    if st.jobs.get(&id).await.is_none() {
        return Err(AppError::not_found(format!("job {id}")));
    }
    let mut rx = st.jobs.subscribe();
    let job_id = id.clone();
    let (tx, rx_ch) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

    tokio::spawn(async move {
        // Initial snapshot event
        let _ = tx
            .send(Ok(Event::default()
                .event("snapshot")
                .data(json!({"job_id": job_id}).to_string())))
            .await;
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let matches = match &ev {
                        JobEvent::Log { job_id: j, .. }
                        | JobEvent::Metric { job_id: j, .. }
                        | JobEvent::Progress { job_id: j, .. }
                        | JobEvent::Status { job_id: j, .. }
                        | JobEvent::Done { job_id: j, .. }
                        | JobEvent::Error { job_id: j, .. } => j == &job_id,
                    };
                    if !matches {
                        continue;
                    }
                    let data = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into());
                    if tx
                        .send(Ok(Event::default().event("job").data(data)))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    if matches!(ev, JobEvent::Done { .. } | JobEvent::Error { .. }) {
                        break;
                    }
                }
                Err(RecvError::Lagged(_)) => continue,
                Err(RecvError::Closed) => break,
            }
        }
    });

    let stream = ReceiverStream::new(rx_ch);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Deserialize)]
struct CryptoBody {
    plaintext: String,
    /// Optional base64/hex 32-byte key; generated if omitted.
    key: Option<String>,
}

async fn crypto_roundtrip(Json(body): Json<CryptoBody>) -> Result<Json<Value>, AppError> {
    let key = if let Some(raw) = body.key.as_deref() {
        resolve_encryption_key(Some(raw)).map_err(|e| AppError::bad_request(e.to_string()))?
    } else {
        generate_encryption_key()
    };
    let enc =
        encrypt_field(&body.plaintext, &key).map_err(|e| AppError::bad_request(e.to_string()))?;
    let dec = decrypt_field(&enc, &key).map_err(|e| AppError::bad_request(e.to_string()))?;
    if dec != body.plaintext {
        return Err(AppError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "roundtrip mismatch".into(),
        });
    }
    Ok(Json(json!({
        "ok": true,
        "ciphertext": enc,
        "plaintext": dec,
        "prefix": seiso_crypto::PREFIX,
    })))
}

#[derive(Deserialize)]
struct SandboxBody {
    /// Path segments under data_dir (e.g. ["models", "alice"]).
    parts: Vec<String>,
    /// Optional relative artifact name check.
    relative_artifact: Option<String>,
}

async fn sandbox_join(
    State(st): State<AppState>,
    Json(body): Json<SandboxBody>,
) -> Result<Json<Value>, AppError> {
    let parts: Vec<&str> = body.parts.iter().map(String::as_str).collect();
    let joined = safe_join(&st.settings.data_dir, &parts)
        .map_err(|e| AppError::bad_request(e.to_string()))?;
    let mut out = json!({
        "path": joined.display().to_string(),
        "data_dir": st.settings.data_dir.display().to_string(),
    });
    if let Some(rel) = body.relative_artifact.as_deref() {
        let name = assert_relative_artifact_name(rel, "relative_artifact")
            .map_err(AppError::bad_request)?;
        out["relative_artifact"] = json!(name);
    }
    Ok(Json(out))
}

#[derive(Debug)]
pub struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }
}

impl From<seiso_db::DbError> for AppError {
    fn from(e: seiso_db::DbError) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: e.to_string(),
        }
    }
}

impl From<seiso_jobs::JobError> for AppError {
    fn from(e: seiso_jobs::JobError) -> Self {
        let status = match &e {
            seiso_jobs::JobError::NotFound(_) => StatusCode::NOT_FOUND,
            seiso_jobs::JobError::Terminal(_) => StatusCode::CONFLICT,
            seiso_jobs::JobError::UnsupportedKind(_) => StatusCode::BAD_REQUEST,
            seiso_jobs::JobError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            message: e.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

/// Helper for tests: wait for job via supervisor.
pub async fn wait_job(
    jobs: &JobSupervisor,
    id: &str,
    timeout: Duration,
) -> Result<JobRecord, seiso_jobs::JobError> {
    jobs.wait_until_terminal(id, timeout).await
}
