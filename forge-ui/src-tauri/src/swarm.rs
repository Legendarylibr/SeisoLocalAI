use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use sysinfo::System;
use uuid::Uuid;

const MAX_CONCURRENT_AGENTS: usize = 4;
const STALL_TIMEOUT_SECS: u64 = 120;
const MAX_AGENT_RUNTIME_SECS: u64 = 600;
const MAX_STRING_LEN: usize = 64 * 1024;
const MAX_GOAL_LEN: usize = 4096;
const MAX_PERSISTED_FILE_BYTES: u64 = 10 * 1024 * 1024;
const WATCHDOG_INTERVAL_SECS: u64 = 15;

// ── Shared state ────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SubagentState {
    pub id: String,
    pub branch: String,
    pub worktree_path: String,
    pub role: String,
    pub status: String,
    pub progress: String,
    pub started_at: Option<String>,
    pub last_activity: Option<String>,
    pub exit_code: Option<i32>,
    pub output_summary: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SwarmRun {
    pub id: String,
    pub goal: String,
    pub preset: String,
    pub started_at: String,
    pub updated_at: String,
    pub status: String,
    pub subagents: Vec<SubagentState>,
    pub merged: bool,
    pub aggregator_note: Option<String>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AgentManifest {
    pub id: String,
    pub role: String,
    pub goal: String,
    pub swarm_run_id: String,
    pub branch: String,
    pub peers: Vec<String>,
    pub worktree_paths: Vec<String>,
}

pub struct SwarmOrchestrator {
    runs: Arc<Mutex<HashMap<String, SwarmRun>>>,
    state_dir: PathBuf,
    system: Arc<Mutex<System>>,
    shutdown_flag: Arc<AtomicBool>,
}

impl SwarmOrchestrator {
    pub fn new(state_dir: PathBuf) -> Self {
        fs::create_dir_all(&state_dir).ok();
        let orchestrator = SwarmOrchestrator {
            runs: Arc::new(Mutex::new(HashMap::new())),
            state_dir,
            system: Arc::new(Mutex::new(System::new())),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
        };
        orchestrator.restore_persisted();
        orchestrator
    }

    /// Resolve a worktree path and require it to be a real directory contained
    /// under the Seiso data directory (parent of the orchestrator's state dir).
    /// Rejects missing paths, `..` escapes, and symlink escapes via
    /// canonicalization. Returns `None` for anything outside the data root.
    fn contained_worktree_path(&self, worktree_path: &str) -> Option<PathBuf> {
        let root = self.state_dir.parent()?.canonicalize().ok()?;
        let path = PathBuf::from(worktree_path);
        if !path.is_absolute() {
            return None;
        }
        let canonical = path.canonicalize().ok()?;
        if canonical.starts_with(&root) {
            Some(canonical)
        } else {
            None
        }
    }

    // ── Subagent awareness ───────────────────────────────────────────────

    pub fn agent_manifest(&self, swarm_run_id: &str, agent_id: &str) -> Option<AgentManifest> {
        let runs = self.runs.lock();
        let run = runs.get(swarm_run_id)?;
        let agent = run.subagents.iter().find(|a| a.id == agent_id)?;
        let peers: Vec<String> = run
            .subagents
            .iter()
            .filter(|a| a.id != agent_id)
            .map(|a| a.id.clone())
            .collect();
        let worktree_paths: Vec<String> = run
            .subagents
            .iter()
            .map(|a| a.worktree_path.clone())
            .collect();
        Some(AgentManifest {
            id: agent.id.clone(),
            role: agent.role.clone(),
            goal: run.goal.clone(),
            swarm_run_id: swarm_run_id.to_string(),
            branch: agent.branch.clone(),
            peers,
            worktree_paths,
        })
    }

    pub fn write_peer_manifests(&self, run_id: &str) {
        let run = self.get_run(run_id);
        let Some(run) = run else { return };
        for agent in &run.subagents {
            if agent.status != "running" {
                continue;
            }
            // Re-validate the stored path before every write — never trust
            // persisted or frontend-supplied paths for file writes.
            let Some(safe) = self.contained_worktree_path(&agent.worktree_path) else {
                continue;
            };
            let manifest = self
                .agent_manifest(run_id, &agent.id)
                .unwrap_or_else(|| panic!("missing agent {} in run {}", agent.id, run_id));
            let json = serde_json::to_string_pretty(&manifest).unwrap_or_default();
            let path = safe.join(".seiso-peer-manifest.json");
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::write(&path, &json).ok();
        }
    }

    // ── Resource limits ──────────────────────────────────────────────────

    pub fn can_spawn(&self) -> bool {
        let runs = self.runs.lock();
        let running_count = runs
            .values()
            .flat_map(|r| &r.subagents)
            .filter(|a| a.status == "running")
            .count();
        if running_count >= MAX_CONCURRENT_AGENTS {
            return false;
        }
        let mut sys = self.system.lock();
        sys.refresh_memory();
        let free_mb = sys.free_memory() / (1024 * 1024);
        free_mb >= 1024
    }

    pub fn spawn_capacity(&self) -> usize {
        let runs = self.runs.lock();
        let running_count = runs
            .values()
            .flat_map(|r| &r.subagents)
            .filter(|a| a.status == "running")
            .count();
        MAX_CONCURRENT_AGENTS.saturating_sub(running_count)
    }

    // ── Lifecycle ────────────────────────────────────────────────────────

    pub fn create_swarm_run(&self, goal: String, preset: String) -> String {
        let id = format!("swarm-{}", &Uuid::new_v4().to_string()[..8]);
        let now = Utc::now().to_rfc3339();
        let goal: String = goal.chars().take(MAX_GOAL_LEN).collect();
        let preset: String = preset.chars().take(MAX_GOAL_LEN).collect();
        let run = SwarmRun {
            id: id.clone(),
            goal,
            preset,
            started_at: now.clone(),
            updated_at: now,
            status: "running".to_string(),
            subagents: Vec::new(),
            merged: false,
            aggregator_note: None,
        };
        {
            let mut runs = self.runs.lock();
            runs.insert(id.clone(), run);
        }
        self.persist_runs();
        id
    }

    pub fn register_subagent(
        &self,
        run_id: &str,
        agent_id: &str,
        branch: &str,
        worktree_path: &str,
        role: &str,
    ) -> Result<(), String> {
        // Validate the worktree path before storing it — it is later used for
        // file writes. Must be a real directory under the Seiso data dir.
        let safe = self
            .contained_worktree_path(worktree_path)
            .ok_or_else(|| {
                "worktree_path must be an existing directory under the Seiso data directory".to_string()
            })?;
        {
            let mut runs = self.runs.lock();
            let run = runs
                .get_mut(run_id)
                .ok_or_else(|| format!("run {} not found", run_id))?;
            if run.subagents.len() >= MAX_CONCURRENT_AGENTS {
                return Err(format!(
                    "run {} already has {} subagents (max {})",
                    run_id,
                    run.subagents.len(),
                    MAX_CONCURRENT_AGENTS
                ));
            }
            let now = Utc::now().to_rfc3339();
            run.subagents.push(SubagentState {
                id: agent_id.to_string(),
                branch: branch.chars().take(MAX_STRING_LEN).collect(),
                worktree_path: safe.to_string_lossy().into_owned(),
                role: role.chars().take(MAX_STRING_LEN).collect(),
                status: "running".to_string(),
                progress: "initializing".to_string(),
                started_at: Some(now.clone()),
                last_activity: Some(now.clone()),
                exit_code: None,
                output_summary: None,
                error: None,
            });
            run.updated_at = now;
        }
        self.write_peer_manifests(run_id);
        self.persist_runs();
        Ok(())
    }

    pub fn update_subagent(
        &self,
        run_id: &str,
        agent_id: &str,
        status: &str,
        progress: &str,
        output: Option<&str>,
        exit_code: Option<i32>,
        error: Option<&str>,
    ) {
        {
            let mut runs = self.runs.lock();
            if let Some(run) = runs.get_mut(run_id) {
                let now = Utc::now().to_rfc3339();
                if let Some(agent) = run.subagents.iter_mut().find(|a| a.id == agent_id) {
                    agent.status = status.chars().take(MAX_STRING_LEN).collect();
                    agent.progress = progress.chars().take(MAX_STRING_LEN).collect();
                    agent.last_activity = Some(now.clone());
                    agent.exit_code = exit_code;
                    if let Some(o) = output {
                        agent.output_summary = Some(o.chars().take(MAX_STRING_LEN).collect());
                    }
                    if let Some(e) = error {
                        agent.error = Some(e.chars().take(MAX_STRING_LEN).collect());
                    }
                }
                run.updated_at = now;

                let terminal: [&str; 3] = ["done", "failed", "stalled"];
                let all_done = run
                    .subagents
                    .iter()
                    .all(|a| terminal.contains(&a.status.as_str()));
                if all_done {
                    let any_failed = run
                        .subagents
                        .iter()
                        .any(|a| a.status == "failed" || a.status == "stalled");
                    run.status = if any_failed {
                        "partial".to_string()
                    } else {
                        "done".to_string()
                    };
                }
            }
        }
        self.persist_runs();
    }

    // ── Aggregation ──────────────────────────────────────────────────────

    pub fn aggregate_results(&self, run_id: &str) -> Option<String> {
        let note = {
            let mut runs = self.runs.lock();
            let run = runs.get_mut(run_id)?;

            let mut parts: Vec<String> = Vec::new();
            for agent in &run.subagents {
                let summary = agent.output_summary.as_deref().unwrap_or("(no output)");
                let status_icon = match agent.status.as_str() {
                    "done" => "[OK]",
                    "failed" => "[FAIL]",
                    "stalled" => "[STALLED]",
                    _ => "[?]",
                };
                parts.push(format!(
                    "{} {} (branch: {}): {}",
                    status_icon, agent.role, agent.branch, summary
                ));
                if let Some(err) = &agent.error {
                    parts.push(format!("  error: {}", err));
                }
            }

            let note = parts.join("\n");
            run.aggregator_note = Some(note.clone());
            run.updated_at = Utc::now().to_rfc3339();
            note
        };
        self.persist_runs();
        Some(note)
    }

    // ── Stall detection and resume injection ─────────────────────────────

    pub fn detect_and_resolve_stalls(&self) -> Vec<(String, String)> {
        let mut stalled: Vec<(String, String)> = Vec::new();
        {
            let mut runs = self.runs.lock();
            let now = Utc::now();
            for run in runs.values_mut() {
                if run.status != "running" {
                    continue;
                }
                for agent in &mut run.subagents {
                    if agent.status != "running" {
                        continue;
                    }
                    let mut stall_reason: Option<String> = None;
                    if let Some(last) = &agent.last_activity {
                        if let Ok(dt) = DateTime::parse_from_rfc3339(last) {
                            let elapsed = (now - dt.with_timezone(&Utc)).num_seconds() as u64;
                            if elapsed >= STALL_TIMEOUT_SECS {
                                stall_reason = Some("stalled: no activity".to_string());
                            }
                        }
                    }
                    // Hard upper bound on runtime — an agent that keeps reporting
                    // (heartbeat) but never finishes must not run forever.
                    if stall_reason.is_none() {
                        if let Some(started) = &agent.started_at {
                            if let Ok(dt) = DateTime::parse_from_rfc3339(started) {
                                let elapsed = (now - dt.with_timezone(&Utc)).num_seconds() as u64;
                                if elapsed >= MAX_AGENT_RUNTIME_SECS {
                                    stall_reason = Some(format!(
                                        "stalled: exceeded max runtime of {}s",
                                        MAX_AGENT_RUNTIME_SECS
                                    ));
                                }
                            }
                        }
                    }
                    if let Some(reason) = stall_reason {
                        agent.status = "stalled".to_string();
                        agent.progress = reason.clone();
                        let agent_id = agent.id.clone();
                        let resume = format!(
                            "You appear to be stalled. Last progress: {}. \
                             Please resume the task and finish it. \
                             If you need to re-read context, do so now. \
                             Do not repeat work already done.",
                            agent.progress
                        );
                        // Re-validate the path before writing the resume prompt.
                        if let Some(safe) = self.contained_worktree_path(&agent.worktree_path) {
                            let resume_path = safe.join(".seiso-resume-prompt.txt");
                            fs::write(&resume_path, &resume).ok();
                            stalled.push((
                                agent_id,
                                format!("resume prompt written to {}", resume_path.display()),
                            ));
                        } else {
                            stalled.push((agent_id, format!("stalled: {}", reason)));
                        }
                    }
                }
            }
        }
        self.persist_runs();
        stalled
    }

    // ── Verification ─────────────────────────────────────────────────────

    pub fn verify_completion(&self, run_id: &str) -> Vec<String> {
        let mut warnings: Vec<String> = Vec::new();
        let runs = self.runs.lock();
        if let Some(run) = runs.get(run_id) {
            for agent in &run.subagents {
                if let Some(output) = &agent.output_summary {
                    let truncation_signals = [
                        "... (truncated",
                        "[TRUNCATED]",
                        "(output clipped)",
                        "... [max tokens]",
                        "MAX_TOKENS_REACHED",
                        "\n...\n...",
                    ];
                    for signal in &truncation_signals {
                        if output.contains(signal) {
                            warnings.push(format!(
                                "agent '{}' ({}) output appears truncated (contains '{}')",
                                agent.id, agent.role, signal
                            ));
                        }
                    }
                    let trimmed = output.trim();
                    if !trimmed.is_empty() {
                        let last_char = trimmed.chars().last().unwrap();
                        if last_char.is_alphanumeric() {
                            warnings.push(format!(
                                "agent '{}' ({}) output ends mid-content \
                                 (last char '{}' not punctuation) — possible truncation",
                                agent.id, agent.role, last_char
                            ));
                        }
                    }
                }
                if agent.status == "failed" {
                    warnings.push(format!(
                        "agent '{}' ({}) failed: {}",
                        agent.id,
                        agent.role,
                        agent.error.as_deref().unwrap_or("unknown error")
                    ));
                }
                if agent.status == "stalled" {
                    warnings.push(format!(
                        "agent '{}' ({}) stalled and was resumed via prompt injection",
                        agent.id, agent.role
                    ));
                }
            }
        }
        warnings
    }

    // ── Watchdog ─────────────────────────────────────────────────────────

    pub fn watchdog_tick(&self) -> Vec<String> {
        let mut actions: Vec<String> = Vec::new();
        let stalled = self.detect_and_resolve_stalls();
        for (agent_id, action) in &stalled {
            actions.push(format!("agent {} stalled: {}", agent_id, action));
        }
        let mut sys = self.system.lock();
        sys.refresh_memory();
        let free_mb = sys.free_memory() / (1024 * 1024);
        if free_mb < 512 {
            actions.push(format!(
                "WARNING: system memory critically low ({} MB free)",
                free_mb
            ));
        }
        actions
    }

    // ── Persistence ──────────────────────────────────────────────────────

    fn persist_runs(&self) {
        let path = self.state_dir.join("swarm_runs.json");
        let runs = self.runs.lock();
        let json = serde_json::to_string_pretty(&*runs).unwrap_or_default();
        // Write to a temp file then rename so a crash mid-write cannot corrupt
        // the persisted state (atomic on POSIX).
        let tmp = self.state_dir.join("swarm_runs.json.tmp");
        if fs::write(&tmp, &json).is_ok() {
            let _ = fs::rename(&tmp, &path);
        }
    }

    fn restore_persisted(&self) {
        let path = self.state_dir.join("swarm_runs.json");
        let Ok(meta) = fs::metadata(&path) else {
            return;
        };
        // Refuse to parse an oversized/crafted file.
        if meta.len() > MAX_PERSISTED_FILE_BYTES {
            return;
        }
        let Ok(data) = fs::read_to_string(&path) else {
            return;
        };
        let Ok(runs) = serde_json::from_str::<HashMap<String, SwarmRun>>(&data) else {
            return;
        };
        let mut current = self.runs.lock();
        for (id, run) in runs {
            if run.status != "running" {
                continue;
            }
            // Do not re-activate runs whose worktree paths are not contained
            // under the data dir — they would feed watchdog file writes.
            let paths_ok = run
                .subagents
                .iter()
                .all(|a| self.contained_worktree_path(&a.worktree_path).is_some());
            if paths_ok && run.subagents.len() <= MAX_CONCURRENT_AGENTS {
                current.insert(id, run);
            }
        }
    }

    // ── Query ────────────────────────────────────────────────────────────

    pub fn get_run(&self, run_id: &str) -> Option<SwarmRun> {
        let runs = self.runs.lock();
        runs.get(run_id).cloned()
    }

    pub fn list_runs(&self) -> Vec<SwarmRun> {
        let runs = self.runs.lock();
        let mut list: Vec<SwarmRun> = runs.values().cloned().collect();
        list.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        list
    }

    pub fn set_merged(&self, run_id: &str) {
        {
            let mut runs = self.runs.lock();
            if let Some(run) = runs.get_mut(run_id) {
                run.merged = true;
                run.updated_at = Utc::now().to_rfc3339();
            }
        }
        self.persist_runs();
    }

    pub fn shutdown_flag(&self) -> Arc<AtomicBool> {
        self.shutdown_flag.clone()
    }

    pub fn state_dir(&self) -> &PathBuf {
        &self.state_dir
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub fn resume_prompt(agent_id: &str, goal: &str) -> String {
    format!(
        r#"[RESUME PROMPT]
Agent {agent_id} appears to have stalled while working on:

  {goal}

Please resume immediately. Do NOT repeat work already completed.
Check your worktree for partial results and finish the remaining tasks.
If something is blocked, state what is missing and continue with what you can.

Complete the task. Do not stop until all acceptance criteria are met.
"#,
    )
}

pub fn verify_output_prompt(agent_id: &str, role: &str) -> String {
    format!(
        r#"[VERIFICATION]
Agent {agent_id} ({role}) has completed its work.

Checking output completeness:
- Does the output end with a clear conclusion or result?
- Are there any truncation markers (..., [TRUNCATED], etc.)?
- Were all required operations completed?
- Are there any syntax errors or incomplete statements?

If anything is incomplete, describe what is missing so the aggregator
can decide whether to resume the agent or proceed with partial results.
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_orchestrator() -> (SwarmOrchestrator, PathBuf) {
        let dir = std::env::temp_dir().join(format!("seiso-swarm-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let wt = dir.join("worktrees");
        fs::create_dir_all(&wt).unwrap();
        (SwarmOrchestrator::new(dir.join("desktop")), wt)
    }

    #[test]
    fn test_can_spawn_under_limit() {
        let (orchestrator, _) = test_orchestrator();
        assert!(orchestrator.can_spawn());
    }

    #[test]
    fn test_create_run_and_register_subagent() {
        let (orchestrator, wt) = test_orchestrator();
        let run_id = orchestrator.create_swarm_run("test goal".to_string(), "pair".to_string());
        orchestrator
            .register_subagent(&run_id, "agent-1", "feat/test", wt.to_str().unwrap(), "worker")
            .unwrap();
        let run = orchestrator.get_run(&run_id).unwrap();
        assert_eq!(run.subagents.len(), 1);
        assert_eq!(run.subagents[0].role, "worker");
    }

    #[test]
    fn test_register_rejects_path_outside_data_dir() {
        let (orchestrator, _) = test_orchestrator();
        let run_id = orchestrator.create_swarm_run("test".to_string(), "pair".to_string());
        // /etc exists and is absolute but is not under the data dir.
        let err = orchestrator
            .register_subagent(&run_id, "a1", "b1", "/etc", "worker")
            .unwrap_err();
        assert!(err.contains("worktree_path"));
    }

    #[test]
    fn test_register_rejects_missing_path() {
        let (orchestrator, _) = test_orchestrator();
        let run_id = orchestrator.create_swarm_run("test".to_string(), "pair".to_string());
        let missing = std::env::temp_dir()
            .join(format!("seiso-missing-{}", Uuid::new_v4()))
            .to_string_lossy()
            .into_owned();
        assert!(orchestrator
            .register_subagent(&run_id, "a1", "b1", &missing, "worker")
            .is_err());
    }

    #[test]
    fn test_register_enforces_agent_cap() {
        let (orchestrator, wt) = test_orchestrator();
        let run_id = orchestrator.create_swarm_run("test".to_string(), "pair".to_string());
        for i in 0..MAX_CONCURRENT_AGENTS {
            let agent_dir = wt.join(format!("agent-{}", i));
            fs::create_dir_all(&agent_dir).unwrap();
            orchestrator
                .register_subagent(
                    &run_id,
                    &format!("a{}", i),
                    "b",
                    agent_dir.to_str().unwrap(),
                    "worker",
                )
                .unwrap();
        }
        let extra_dir = wt.join("extra");
        fs::create_dir_all(&extra_dir).unwrap();
        assert!(orchestrator
            .register_subagent(&run_id, "overflow", "b", extra_dir.to_str().unwrap(), "worker")
            .is_err());
    }

    #[test]
    fn test_aggregate_results() {
        let (orchestrator, wt) = test_orchestrator();
        let run_id = orchestrator.create_swarm_run("test".to_string(), "pair".to_string());
        let wt1 = wt.join("wt1");
        let wt2 = wt.join("wt2");
        fs::create_dir_all(&wt1).unwrap();
        fs::create_dir_all(&wt2).unwrap();
        orchestrator
            .register_subagent(&run_id, "a1", "b1", wt1.to_str().unwrap(), "worker")
            .unwrap();
        orchestrator
            .register_subagent(&run_id, "a2", "b2", wt2.to_str().unwrap(), "completion")
            .unwrap();
        orchestrator.update_subagent(
            &run_id,
            "a1",
            "done",
            "complete",
            Some("worked"),
            Some(0),
            None,
        );
        orchestrator.update_subagent(
            &run_id,
            "a2",
            "done",
            "complete",
            Some("verified"),
            Some(0),
            None,
        );
        let summary = orchestrator.aggregate_results(&run_id);
        assert!(summary.is_some());
        assert!(summary.unwrap().contains("[OK] worker"));
    }
}
