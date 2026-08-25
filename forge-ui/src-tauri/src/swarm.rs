use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use sysinfo::{Pid, PidExt, ProcessExt, System, SystemExt};
use uuid::Uuid;

const MAX_CONCURRENT_AGENTS: usize = 4;
const STALL_TIMEOUT_SECS: u64 = 120;
const MAX_AGENT_RUNTIME_SECS: u64 = 600;
const MIN_VRAM_MB: u64 = 2048;
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
        let worktree_paths: Vec<String> =
            run.subagents.iter().map(|a| a.worktree_path.clone()).collect();
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

    pub fn write_peer_manifests(&self, run: &SwarmRun) {
        let rid = run.id.clone();
        for agent in &run.subagents {
            if agent.status != "running" {
                continue;
            }
            let manifest = self
                .agent_manifest(&rid, &agent.id)
                .unwrap_or_else(|| panic!("missing agent {} in run {}", agent.id, rid));
            let json = serde_json::to_string_pretty(&manifest).unwrap_or_default();
            let path = PathBuf::from(&agent.worktree_path).join(".seiso-peer-manifest.json");
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
        let mut runs = self.runs.lock();
        runs.insert(id.clone(), run);
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
        let mut runs = self.runs.lock();
        let run = runs
            .get_mut(run_id)
            .ok_or_else(|| format!("run {} not found", run_id))?;
        let now = Utc::now().to_rfc3339();
        run.subagents.push(SubagentState {
            id: agent_id.to_string(),
            branch: branch.to_string(),
            worktree_path: worktree_path.to_string(),
            role: role.to_string(),
            status: "running".to_string(),
            progress: "initializing".to_string(),
            started_at: Some(now.clone()),
            last_activity: Some(now),
            exit_code: None,
            output_summary: None,
            error: None,
        });
        run.updated_at = now;
        self.write_peer_manifests(run);
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
        let mut runs = self.runs.lock();
        if let Some(run) = runs.get_mut(run_id) {
            let now = Utc::now().to_rfc3339();
            if let Some(agent) = run.subagents.iter_mut().find(|a| a.id == agent_id) {
                agent.status = status.to_string();
                agent.progress = progress.to_string();
                agent.last_activity = Some(now.clone());
                agent.exit_code = exit_code;
                if let Some(o) = output {
                    agent.output_summary = Some(o.to_string());
                }
                if let Some(e) = error {
                    agent.error = Some(e.to_string());
                }
            }
            run.updated_at = now;

            let terminal: [&str; 3] = ["done", "failed", "stalled"];
            let all_done = run.subagents.iter().all(|a| terminal.contains(&a.status.as_str()));
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
            self.persist_runs();
        }
    }

    // ── Aggregation ──────────────────────────────────────────────────────

    pub fn aggregate_results(&self, run_id: &str) -> Option<String> {
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
            if let Some(ref err) = agent.error {
                parts.push(format!("  error: {}", err));
            }
        }

        let note = parts.join("\n");
        run.aggregator_note = Some(note.clone());
        run.updated_at = Utc::now().to_rfc3339();
        self.persist_runs();
        Some(note)
    }

    // ── Stall detection and resume injection ─────────────────────────────

    pub fn detect_and_resolve_stalls(&self) -> Vec<(String, String)> {
        let mut stalled: Vec<(String, String)> = Vec::new();
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
                if let Some(ref last) = agent.last_activity {
                    if let Ok(dt) = DateTime::parse_from_rfc3339(last) {
                        let elapsed = (now - dt).num_seconds() as u64;
                        if elapsed < STALL_TIMEOUT_SECS {
                            continue;
                        }
                        agent.status = "stalled".to_string();
                        agent.progress = "stalled: no activity".to_string();
                        let agent_id = agent.id.clone();
                        let wt_path = agent.worktree_path.clone();
                        let resume = format!(
                            "You appear to be stalled. Last progress: {}. \
                             Please resume the task and finish it. \
                             If you need to re-read context, do so now. \
                             Do not repeat work already done.",
                            agent.progress
                        );
                        let resume_path = PathBuf::from(&wt_path).join(".seiso-resume-prompt.txt");
                        fs::write(&resume_path, &resume).ok();
                        stalled.push((
                            agent_id,
                            format!("resume prompt written to {}", resume_path.display()),
                        ));
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
                if let Some(ref output) = agent.output_summary {
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
        fs::write(&path, &json).ok();
    }

    fn restore_persisted(&self) {
        let path = self.state_dir.join("swarm_runs.json");
        if let Ok(data) = fs::read_to_string(&path) {
            if let Ok(runs) = serde_json::from_str::<HashMap<String, SwarmRun>>(&data) {
                let mut current = self.runs.lock();
                for (id, run) in runs {
                    if run.status == "running" {
                        current.insert(id, run);
                    }
                }
            }
        }
    }

    // ── Query ────────────────────────────────────────────────────────────

    pub fn get_run(&self, run_id: &str) -> Option<SwarmRun> {
        let runs = self.runs.lock();
        runs.get(run_id).cloned()
    }

    pub fn list_runs(&self) -> Vec<SwarmRun> {
        let mut runs = self.runs.lock();
        let mut list: Vec<SwarmRun> = runs.values().cloned().collect();
        list.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        list
    }

    pub fn set_merged(&self, run_id: &str) {
        let mut runs = self.runs.lock();
        if let Some(run) = runs.get_mut(run_id) {
            run.merged = true;
            run.updated_at = Utc::now().to_rfc3339();
            self.persist_runs();
        }
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

    #[test]
    fn test_can_spawn_under_limit() {
        let orchestrator = SwarmOrchestrator::new(PathBuf::from("/tmp/.seiso-test"));
        assert!(orchestrator.can_spawn());
    }

    #[test]
    fn test_create_run_and_register_subagent() {
        let orchestrator = SwarmOrchestrator::new(PathBuf::from("/tmp/.seiso-test"));
        let run_id = orchestrator.create_swarm_run("test goal".to_string(), "pair".to_string());
        orchestrator
            .register_subagent(&run_id, "agent-1", "feat/test", "/tmp/wt", "worker")
            .unwrap();
        let run = orchestrator.get_run(&run_id).unwrap();
        assert_eq!(run.subagents.len(), 1);
        assert_eq!(run.subagents[0].role, "worker");
    }

    #[test]
    fn test_aggregate_results() {
        let orchestrator = SwarmOrchestrator::new(PathBuf::from("/tmp/.seiso-test"));
        let run_id = orchestrator.create_swarm_run("test".to_string(), "pair".to_string());
        orchestrator
            .register_subagent(&run_id, "a1", "b1", "/tmp/wt1", "worker")
            .unwrap();
        orchestrator
            .register_subagent(&run_id, "a2", "b2", "/tmp/wt2", "completion")
            .unwrap();
        orchestrator.update_subagent(&run_id, "a1", "done", "complete", Some("worked"), Some(0), None);
        orchestrator
            .update_subagent(&run_id, "a2", "done", "complete", Some("verified"), Some(0), None);
        let summary = orchestrator.aggregate_results(&run_id);
        assert!(summary.is_some());
        assert!(summary.unwrap().contains("[OK] worker"));
    }
}
