"""Forge server and doctor commands."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from seiso_cli.console import console


def forge(
    host: str | None = typer.Option(None, help="Bind host (default: 127.0.0.1)"),
    port: int | None = typer.Option(None, help="Port (default: 8765)"),
    reload: bool = typer.Option(False, help="Dev auto-reload"),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open Forge in the system browser when /health is ready",
    ),
) -> None:
    """Launch Seiso web server (UI + API)."""
    import os

    import uvicorn

    from forge.config import get_settings
    from forge.instance_lock import ForgeAlreadyRunningError, acquire_forge_instance_locks
    from forge.launch import schedule_browser_open

    settings = get_settings()
    bind_host = host or settings.host
    bind_port = port or settings.port
    try:
        instance_locks = acquire_forge_instance_locks(
            host=bind_host,
            port=bind_port,
            data_dir=settings.data_dir,
        )
    except ForgeAlreadyRunningError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    forge_url = f"http://{bind_host}:{bind_port}"
    should_open = open_browser or os.environ.get("SEISO_OPEN_BROWSER", "").strip() in {
        "1",
        "true",
        "yes",
    }
    if should_open:
        schedule_browser_open(forge_url)
    console.print(f"[bold green]Seiso[/] → {forge_url}")
    try:
        uvicorn.run(
            "forge.main:create_app",
            factory=True,
            host=bind_host,
            port=bind_port,
            reload=reload,
            log_level="info",
            proxy_headers=settings.trust_proxy,
            forwarded_allow_ips="127.0.0.1,::1" if settings.trust_proxy else None,
        )
    finally:
        instance_locks.release()


def doctor(
    network: bool = typer.Option(False, "--network", help="Also probe huggingface.co reachability"),
) -> None:
    """Diagnose install, runtime, and Hugging Face setup."""
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "doctor.sh"
    if not script.is_file():
        console.print(f"[red]Doctor script not found:[/] {script}")
        raise typer.Exit(1)
    args = [str(script)]
    if network:
        args.append("--network")
    raise typer.Exit(subprocess.call(args))