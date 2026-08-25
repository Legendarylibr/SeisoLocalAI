use crate::swarm::{AgentManifest, SwarmOrchestrator, SwarmRun};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;

#[derive(Serialize, Deserialize)]
pub struct BackendStatus {
    pub running: bool,
    pub port: u16,
    pub pid: Option<u32>,
    pub ready: bool,
}

#[derive(Serialize, Deserialize)]
pub struct SwarmSummary {
    pub id: String,
    pub goal: String,
    pub status: String,
    pub preset: String,
    pub started_at: String,
    pub updated_at: String,
    pub agent_count: usize,
    pub merged: bool,
    pub aggregator_note: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ResourceInfo {
    pub free_memory_mb: u64,
    pub total_memory_mb: u64,
    pub spawn_capacity: usize,
    pub running_agents: usize,
}

impl From<SwarmRun> for SwarmSummary {
    fn from(run: SwarmRun) -> Self {
        let agent_count = run.subagents.len();
        SwarmSummary {
            id: run.id,
            goal: run.goal,
            status: run.status,
            preset: run.preset,
            started_at: run.started_at,
            updated_at: run.updated_at,
            agent_count,
            merged: run.merged,
            aggregator_note: run.aggregator_note,
        }
    }
}

// ── Backend status ──────────────────────────────────────────────────────────

#[tauri::command]
pub fn get_backend_status() -> BackendStatus {
    BackendStatus {
        running: true,
        port: 8765,
        pid: None,
        ready: true,
    }
}

// ── Swarm lifecycle ─────────────────────────────────────────────────────────

#[tauri::command]
pub fn create_swarm_run(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
    goal: String,
    preset: String,
) -> Result<String, String> {
    let orchestrator = state.lock();
    Ok(orchestrator.create_swarm_run(goal, preset))
}

#[tauri::command]
pub fn register_subagent(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
    run_id: String,
    agent_id: String,
    branch: String,
    worktree_path: String,
    role: String,
) -> Result<(), String> {
    let orchestrator = state.lock();
    orchestrator.register_subagent(&run_id, &agent_id, &branch, &worktree_path, &role)
}

#[tauri::command]
pub fn update_subagent(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
    run_id: String,
    agent_id: String,
    status: String,
    progress: String,
    output: Option<String>,
    exit_code: Option<i32>,
    error: Option<String>,
) {
    let orchestrator = state.lock();
    orchestrator.update_subagent(
        &run_id,
        &agent_id,
        &status,
        &progress,
        output.as_deref(),
        exit_code,
        error.as_deref(),
    );
}

#[tauri::command]
pub fn aggregate_swarm_results(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
    run_id: String,
) -> Result<Option<String>, String> {
    let orchestrator = state.lock();
    Ok(orchestrator.aggregate_results(&run_id))
}

#[tauri::command]
pub fn verify_swarm_completion(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
    run_id: String,
) -> Result<Vec<String>, String> {
    let orchestrator = state.lock();
    Ok(orchestrator.verify_completion(&run_id))
}

#[tauri::command]
pub fn list_swarm_runs(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
) -> Vec<SwarmSummary> {
    let orchestrator = state.lock();
    orchestrator
        .list_runs()
        .into_iter()
        .map(SwarmSummary::from)
        .collect()
}

#[tauri::command]
pub fn get_swarm_run(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
    run_id: String,
) -> Option<SwarmRun> {
    let orchestrator = state.lock();
    orchestrator.get_run(&run_id)
}

#[tauri::command]
pub fn get_agent_manifest(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
    run_id: String,
    agent_id: String,
) -> Option<AgentManifest> {
    let orchestrator = state.lock();
    orchestrator.agent_manifest(&run_id, &agent_id)
}

#[tauri::command]
pub fn can_spawn_agent(state: State<'_, Arc<Mutex<SwarmOrchestrator>>>) -> bool {
    let orchestrator = state.lock();
    orchestrator.can_spawn()
}

#[tauri::command]
pub fn get_resource_info(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
) -> ResourceInfo {
    let orchestrator = state.lock();
    let spawn_capacity = orchestrator.spawn_capacity();
    let running_agents = orchestrator
        .list_runs()
        .iter()
        .flat_map(|r| &r.subagents)
        .filter(|a| a.status == "running")
        .count();
    ResourceInfo {
        free_memory_mb: 0,
        total_memory_mb: 0,
        spawn_capacity,
        running_agents,
    }
}

#[tauri::command]
pub fn watchdog_tick(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
) -> Vec<String> {
    let orchestrator = state.lock();
    orchestrator.watchdog_tick()
}

#[tauri::command]
pub fn set_worktree_merged(
    state: State<'_, Arc<Mutex<SwarmOrchestrator>>>,
    run_id: String,
) {
    let orchestrator = state.lock();
    orchestrator.set_merged(&run_id);
}

// ── Tool availability checks ────────────────────────────────────────────────

#[tauri::command]
pub fn check_ssh_agent() -> bool {
    std::env::var("SSH_AUTH_SOCK").is_ok()
}

#[tauri::command]
pub fn check_github_cli() -> bool {
    std::process::Command::new("gh")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tauri::command]
pub fn check_git_available() -> bool {
    std::process::Command::new("git")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
