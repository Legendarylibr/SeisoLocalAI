//! `seiso-forge` binary entrypoint.

use seiso_core::ForgeSettings;
use tracing::Level;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,seiso_forge=debug")),
        )
        .with_max_level(Level::DEBUG)
        .init();

    let settings = ForgeSettings::from_env()?;
    tracing::info!(
        data_dir = %settings.data_dir.display(),
        bind = %settings.bind_addr(),
        "starting seiso-forge (rust control plane)"
    );
    seiso_forge::serve(settings).await
}
