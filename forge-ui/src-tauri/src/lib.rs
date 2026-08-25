use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::{
    AppHandle, Emitter, Manager,
    menu::{MenuBuilder, MenuItemBuilder},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
};

mod commands;
mod swarm;

/// Backend-sidecar child process handle.
struct SidecarHandle {
    child: Option<tauri_plugin_shell::process::CommandChild>,
}

/// Periodic watchdog ticker.
fn start_watchdog(
    handle: AppHandle,
    orchestrator: Arc<Mutex<swarm::SwarmOrchestrator>>,
) {
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(15));
        loop {
            interval.tick().await;
            let actions = {
                let orch = orchestrator.lock();
                orch.watchdog_tick()
            };
            for action in &actions {
                let _ = handle.emit("watchdog-action", action);
            }
            // Heartbeat
            let _ = handle.emit("backend-heartbeat", serde_json::json!({"ts": chrono::Utc::now().to_rfc3339()}));
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarHandle { child: None })
        .setup(|app| {
            // ── Orchestrator ──────────────────────────────────────────────
            let data_dir = {
                let home = dirs_or_fallback();
                let base = std::env::var("SEISO_DATA_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| home.join(".seiso"));
                base.join("desktop")
            };
            let orchestrator = Arc::new(Mutex::new(swarm::SwarmOrchestrator::new(
                data_dir,
            )));
            app.manage(orchestrator.clone());

            // ── Sidecar ───────────────────────────────────────────────────
            spawn_sidecar(app);

            // ── Tray ──────────────────────────────────────────────────────
            build_tray(app)?;

            // ── Watchdog ──────────────────────────────────────────────────
            start_watchdog(app.handle().clone(), orchestrator);

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_backend_status,
            commands::create_swarm_run,
            commands::register_subagent,
            commands::update_subagent,
            commands::aggregate_swarm_results,
            commands::verify_swarm_completion,
            commands::list_swarm_runs,
            commands::get_swarm_run,
            commands::get_agent_manifest,
            commands::can_spawn_agent,
            commands::get_resource_info,
            commands::watchdog_tick,
            commands::set_worktree_merged,
            commands::check_ssh_agent,
            commands::check_github_cli,
            commands::check_git_available,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Seiso Forge");
}

fn spawn_sidecar(app: &tauri::App) {
    let handle = app.handle().clone();
    let sidecar_state: tauri::State<'_, SidecarHandle> = app.state();

    // Forward SSH_AUTH_SOCK so git operations over SSH work
    let extra_env: Vec<(String, String)> = {
        let mut env = Vec::new();
        if let Ok(sock) = std::env::var("SSH_AUTH_SOCK") {
            env.push(("SSH_AUTH_SOCK".to_string(), sock));
        }
        if let Ok(known) = std::env::var("SSH_KNOWN_HOSTS") {
            env.push(("SSH_KNOWN_HOSTS".to_string(), known));
        }
        env
    };

    let sidecar_cmd = app
        .shell()
        .sidecar("binaries/seiso-sidecar")
        .map_err(|e| {
            eprintln!("sidecar binary not found: {e}");
        });

    if let Ok(cmd) = sidecar_cmd {
        let cmd = extra_env
            .into_iter()
            .fold(cmd, |c, (k, v)| c.env(k, v));

        match cmd.spawn() {
            Ok((mut rx, child)) => {
                *sidecar_state.child.lock() = Some(child);

                tauri::async_runtime::spawn(async move {
                    use tauri_plugin_shell::process::CommandEvent;
                    while let Some(event) = rx.recv().await {
                        match event {
                            CommandEvent::Stdout(line) => {
                                let _ = handle.emit("backend-log", line);
                            }
                            CommandEvent::Stderr(line) => {
                                let _ = handle.emit("backend-log", line);
                            }
                            CommandEvent::Terminated(payload) => {
                                let _ = handle.emit("backend-exited", payload);
                                break;
                            }
                            _ => {}
                        }
                    }
                });
            }
            Err(e) => {
                eprintln!("failed to spawn sidecar: {e}");
            }
        }
    }
}

fn build_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let quit = MenuItemBuilder::with_id("quit", "Quit Seiso").build(app)?;
    let show = MenuItemBuilder::with_id("show", "Show Window").build(app)?;
    let menu = MenuBuilder::new(app)
        .item(&show)
        .separator()
        .item(&quit)
        .build()?;

    TrayIconBuilder::new()
        .menu(&menu)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "quit" => {
                kill_sidecar(app);
                app.exit(0);
            }
            "show" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .build(app)?;

    Ok(())
}

fn kill_sidecar(app: &AppHandle) {
    if let Some(state) = app.try_state::<SidecarHandle>() {
        let mut guard = state.child.lock();
        if let Some(child) = guard.take() {
            let _ = child.kill();
        }
    }
}

fn dirs_or_fallback() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
    } else {
        PathBuf::from(".")
    }
}
