//! Seiso Forge — Rust control plane (Phase 1 scaffold).
//!
//! Listens on `SEISO_HOST`:`SEISO_PORT` (default 127.0.0.1:8765).
//! Python Forge remains default for production until cutover; run this binary
//! when `SEISO_FORGE_IMPL=rust` or for development.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use seiso_core::ForgeSettings;
use seiso_db::Db;
use seiso_jobs::{JobRecord, JobSupervisor};
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;

#[derive(Clone)]
struct AppState {
    settings: Arc<ForgeSettings>,
    db: Db,
    jobs: JobSupervisor,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,seiso_forge=debug")),
        )
        .with_max_level(Level::DEBUG)
        .init();

    let settings = Arc::new(ForgeSettings::from_env()?);
    info!(
        data_dir = %settings.data_dir.display(),
        bind = %settings.bind_addr(),
        "starting seiso-forge (rust control plane)"
    );

    let db = Db::connect(&settings.forge_db_path()).await?;
    let jobs = JobSupervisor::new(settings.data_dir.clone());
    let state = AppState {
        settings: settings.clone(),
        db,
        jobs,
    };

    let api = Router::new()
        .route("/api/health", get(health))
        .route("/api/system", get(system))
        .route("/api/jobs", get(list_jobs).post(create_job))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/jobs/{id}/cancel", post(cancel_job))
        .with_state(state);

    let app = if let Some(dist) = settings.ui_dist.as_ref() {
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
    };

    let addr: SocketAddr = settings.bind_addr().parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(%addr, "listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal");
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
    if body.kind != "train" {
        return Err(AppError::bad_request(format!(
            "unsupported job kind '{}' (phase 1 supports train)",
            body.kind
        )));
    }
    let job = st
        .jobs
        .start_train(body.config)
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

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    fn not_found(msg: impl Into<String>) -> Self {
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
