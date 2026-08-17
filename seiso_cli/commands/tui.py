"""Seiso terminal UI — Forge layout, live Hub, no browser."""

from __future__ import annotations

import sys

import typer

from seiso_cli.console import console


def _reattach_controlling_tty() -> None:
    """Recover a real terminal after ``curl | bash`` (stdin is a pipe)."""
    if sys.stdin.isatty():
        return
    try:
        tty = open("/dev/tty")  # noqa: SIM115
    except OSError:
        return
    sys.stdin = tty


def tui(
    model: str = typer.Option("", "--model", "-m", help="GGUF path, index, or name substring"),
    list_models: bool = typer.Option(False, "--list", help="Print local GGUF files and exit"),
    data_dir: str = typer.Option("", "--data-dir", help="Override SEISO_DATA_DIR"),
) -> None:
    """Local offline UI that mimics Forge in the terminal (no browser).

    Arrow keys scroll, Tab or ←→ switch sidebar/page, Enter selects.
    Same Nostr account as the web UI (recovery key + persisted session).
    """
    from pathlib import Path

    from seiso.security import resolve_data_dir
    from seiso.tui.app import run_tui
    from seiso.tui.offline import discover_local_gguf, format_size, resolve_model_choice

    root = resolve_data_dir(data_dir or None)
    models = discover_local_gguf(root)
    if list_models:
        if not models:
            console.print("No local GGUF files under models/, exports/, or hf_cache/.")
            raise typer.Exit(0)
        for index, item in enumerate(models, start=1):
            console.print(f"  {index:>2}  {item.label}  {format_size(item.size_bytes)}")
        raise typer.Exit(0)

    selected, error = resolve_model_choice(model, models)
    if model and error:
        console.print(f"[red]{error}[/]")
        raise typer.Exit(1)

    _reattach_controlling_tty()
    run_tui(
        data_dir=root,
        initial_model=str(selected.path) if selected else "",
        console=console,
        repo_root=Path(__file__).resolve().parents[2],
    )
