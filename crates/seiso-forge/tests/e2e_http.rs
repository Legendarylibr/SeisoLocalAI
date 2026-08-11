//! End-to-end HTTP tests against a live `seiso-forge` instance + Python worker.

use std::path::PathBuf;
use std::time::Duration;

use seiso_core::ForgeSettings;
use seiso_forge::{app_state_from_settings, build_router};
use serde_json::{json, Value};
use tokio::net::TcpListener;

async fn spawn_forge() -> (String, PathBuf) {
    let data = tempfile::tempdir().expect("tempdir").keep();
    let settings = ForgeSettings {
        host: "127.0.0.1".into(),
        port: 0, // unused — we bind ephemeral
        data_dir: data.clone(),
        localhost_only: true,
        ui_dist: None,
    };
    let state = app_state_from_settings(settings).await.expect("app state");
    let app = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (format!("http://{addr}"), data)
}

async fn get_json(client: &reqwest::Client, url: &str) -> Value {
    client
        .get(url)
        .send()
        .await
        .expect("get")
        .error_for_status()
        .expect("status")
        .json()
        .await
        .expect("json")
}

async fn post_json(client: &reqwest::Client, url: &str, body: Value) -> (u16, Value) {
    let resp = client.post(url).json(&body).send().await.expect("post");
    let status = resp.status().as_u16();
    let v = resp.json().await.expect("json");
    (status, v)
}

async fn poll_job(client: &reqwest::Client, base: &str, id: &str) -> Value {
    let mut final_job = Value::Null;
    for _ in 0..200 {
        final_job = get_json(client, &format!("{base}/api/jobs/{id}")).await;
        let st = final_job["status"].as_str().unwrap_or("");
        if matches!(st, "succeeded" | "failed" | "cancelled") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    final_job
}

#[tokio::test]
async fn e2e_health_and_system() {
    let (base, _data) = spawn_forge().await;
    let client = reqwest::Client::new();
    let health = get_json(&client, &format!("{base}/api/health")).await;
    assert_eq!(health["status"], "ok");
    assert_eq!(health["impl"], "rust");
    assert_eq!(health["schema_version"], "1");

    let system = get_json(&client, &format!("{base}/api/system")).await;
    assert_eq!(system["forge_impl"], "rust");
    assert_eq!(system["features"]["jobs"], true);
    assert_eq!(system["features"]["sse"], true);
}

#[tokio::test]
async fn e2e_train_job_completes() {
    let (base, _data) = spawn_forge().await;
    let client = reqwest::Client::new();

    let (status, created) = post_json(
        &client,
        &format!("{base}/api/jobs"),
        json!({"kind": "train", "config": {"smoke_only": true}}),
    )
    .await;
    assert_eq!(status, 202, "body={created}");
    let id = created["id"].as_str().expect("id").to_string();
    assert_eq!(created["kind"], "train");

    let final_job = poll_job(&client, &base, &id).await;
    assert_eq!(final_job["status"], "succeeded", "job={final_job}");
    let logs = final_job["logs"].as_array().cloned().unwrap_or_default();
    assert!(
        logs.iter()
            .any(|l| l.as_str().unwrap_or("").contains("smoke train complete")),
        "logs={logs:?}"
    );

    let listed = get_json(&client, &format!("{base}/api/jobs")).await;
    let jobs = listed["jobs"].as_array().expect("jobs arr");
    assert!(jobs.iter().any(|j| j["id"] == id));
}

#[tokio::test]
async fn e2e_export_job_writes_artifact() {
    let (base, data) = spawn_forge().await;
    let client = reqwest::Client::new();

    let (status, created) = post_json(
        &client,
        &format!("{base}/api/jobs"),
        json!({"kind": "export", "config": {"smoke_only": true, "format": "gguf"}}),
    )
    .await;
    assert_eq!(status, 202, "body={created}");
    let id = created["id"].as_str().expect("id").to_string();

    let final_job = poll_job(&client, &base, &id).await;
    assert_eq!(final_job["status"], "succeeded", "job={final_job}");
    let arts = final_job["artifacts"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(!arts.is_empty(), "artifacts empty: {final_job}");
    let art_path = arts[0].as_str().unwrap();
    assert!(
        PathBuf::from(art_path).is_file(),
        "artifact missing: {art_path} data={}",
        data.display()
    );
}

#[tokio::test]
async fn e2e_crypto_roundtrip_and_sandbox() {
    let (base, _data) = spawn_forge().await;
    let client = reqwest::Client::new();

    let (status, crypto) = post_json(
        &client,
        &format!("{base}/api/crypto/roundtrip"),
        json!({"plaintext": "hello-e2e-secret"}),
    )
    .await;
    assert_eq!(status, 200, "{crypto}");
    assert_eq!(crypto["ok"], true);
    assert!(crypto["ciphertext"]
        .as_str()
        .unwrap()
        .starts_with("enc:v1:"));
    assert_eq!(crypto["plaintext"], "hello-e2e-secret");

    let key_b64 = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
    let (status, crypto2) = post_json(
        &client,
        &format!("{base}/api/crypto/roundtrip"),
        json!({
            "plaintext": "hello-seiso-secret",
            "key": key_b64,
        }),
    )
    .await;
    assert_eq!(status, 200, "{crypto2}");

    let (status, path) = post_json(
        &client,
        &format!("{base}/api/sandbox/join"),
        json!({"parts": ["models", "alice"], "relative_artifact": "checkpoint-best"}),
    )
    .await;
    assert_eq!(status, 200, "{path}");
    assert!(path["path"].as_str().unwrap().contains("models"));
    assert_eq!(path["relative_artifact"], "checkpoint-best");

    let (status, err) = post_json(
        &client,
        &format!("{base}/api/sandbox/join"),
        json!({"parts": [".."]}),
    )
    .await;
    assert_eq!(status, 400, "{err}");
    assert!(err.get("error").is_some());
}

#[tokio::test]
async fn e2e_unknown_job_kind_rejected() {
    let (base, _data) = spawn_forge().await;
    let client = reqwest::Client::new();
    let (status, body) = post_json(
        &client,
        &format!("{base}/api/jobs"),
        json!({"kind": "teleport", "config": {}}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
}
