//! `seiso-rs` — Rust CLI entry (coexists with Python `seiso` until cutover).

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use seiso_core::{resolve_data_dir, ForgeImpl, ForgeSettings, FORGE_IMPL_ENV};
use seiso_sandbox::safe_join;

#[derive(Parser, Debug)]
#[command(
    name = "seiso-rs",
    version,
    about = "Seiso Local AI (Rust control plane CLI)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Print environment, data dir, and sandbox smoke checks
    Doctor {
        /// Also print network-related hints (placeholder)
        #[arg(long)]
        network: bool,
    },
    /// Run the Rust Forge HTTP server (or delegate to Python)
    Forge {
        /// Force implementation: rust | python
        #[arg(long, env = "SEISO_FORGE_IMPL")]
        r#impl: Option<String>,
    },
    /// Show resolved paths for a user-scoped category
    Paths {
        #[arg(long, default_value = "models")]
        category: String,
        #[arg(long, default_value = "local")]
        user_id: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Doctor { network } => doctor(network),
        Commands::Forge { r#impl } => forge(r#impl),
        Commands::Paths { category, user_id } => paths(&category, &user_id),
    }
}

fn doctor(network: bool) -> Result<()> {
    let data = resolve_data_dir(None)?;
    let settings = ForgeSettings::from_env()?;
    println!("Seiso doctor (Rust control plane)");
    println!("  data_dir:     {}", data.display());
    println!("  forge_db:     {}", settings.forge_db_path().display());
    println!("  bind:         {}", settings.bind_addr());
    println!("  forge_impl:   {:?}", ForgeImpl::from_env());
    println!("  ui_dist:      {:?}", settings.ui_dist);
    println!("  SEISO_FORGE_IMPL env key: {FORGE_IMPL_ENV}");
    // Sandbox smoke
    let joined =
        safe_join(&data, &["models", "doctor-check"]).context("safe_join models/doctor-check")?;
    println!("  sandbox ok:   {}", joined.display());
    if network {
        println!("  network:      (phase 1) use `seiso doctor --network` Python for full checks");
    }
    println!("ok");
    Ok(())
}

fn forge(impl_override: Option<String>) -> Result<()> {
    let which = impl_override
        .as_deref()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| match ForgeImpl::from_env() {
            ForgeImpl::Rust => "rust".into(),
            ForgeImpl::Python => "python".into(),
        });
    match which.as_str() {
        "rust" | "rs" => {
            // Prefer cargo-built binary next to this one, else `seiso-forge` on PATH,
            // else `cargo run -p seiso-forge`.
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    let candidate = dir.join("seiso-forge");
                    if candidate.is_file() {
                        let status = Command::new(candidate).status()?;
                        if !status.success() {
                            bail!("seiso-forge exited with {status}");
                        }
                        return Ok(());
                    }
                }
            }
            let status = Command::new("seiso-forge").status().or_else(|_| {
                Command::new("cargo")
                    .args(["run", "-p", "seiso-forge", "--release"])
                    .status()
            })?;
            if !status.success() {
                bail!("forge failed: {status}");
            }
            Ok(())
        }
        "python" | "py" => {
            let status = Command::new("seiso")
                .arg("forge")
                .status()
                .context("failed to spawn python seiso forge — is the venv active?")?;
            if !status.success() {
                bail!("python seiso forge exited {status}");
            }
            Ok(())
        }
        other => bail!("unknown forge impl {other:?} (use rust|python)"),
    }
}

fn paths(category: &str, user_id: &str) -> Result<()> {
    let data = resolve_data_dir(None)?;
    let p = safe_join(&data, &[category, user_id])?;
    println!("{}", p.display());
    // Ensure parent exists for UX
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = PathBuf::from(&p);
    Ok(())
}
