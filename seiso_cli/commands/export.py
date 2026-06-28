"""Export command."""

from __future__ import annotations

from pathlib import Path

import typer

from seiso_cli.console import console


def export_cmd(
    checkpoint: str = typer.Option(..., help="Checkpoint directory"),
    formats: str = typer.Option(
        "merged", help="Comma-separated: merged,lora,full,gguf"
    ),
    profile: str | None = typer.Option(
        None, help="Export profile: lora_bundle, full_bundle, inference, ..."
    ),
    hub_repo: str | None = typer.Option(None, help="Hugging Face repo to push"),
    precheck_only: bool = typer.Option(
        False, help="Run Hub precheck without exporting"
    ),
) -> None:
    """Export checkpoint to merged/GGUF/LoRA/full fine-tune."""
    from forge.config import get_settings
    from seiso.export.pipeline import prepare_export, profile_catalog, run_export_plan

    settings = get_settings()
    ckpt = Path(checkpoint)
    fmt_list = [f.strip() for f in formats.split(",")] if not profile else None

    if profile and profile == "list":
        for entry in profile_catalog():
            console.print(f"  [cyan]{entry['id']}[/] → {', '.join(entry['formats'])}")
        return

    plan = prepare_export(
        checkpoint=ckpt,
        output_dir=settings.exports_dir / ckpt.name,
        formats=fmt_list,
        profile=profile,
        hub_repo=hub_repo,
        hub_token=settings.hf_token or None,
        on_log=lambda m: console.print(m),
    )

    if precheck_only:
        if plan.precheck:
            console.print(plan.precheck.to_dict())
        else:
            console.print("No Hub precheck requested (set --hub-repo)")
        return

    results = run_export_plan(
        plan, hub_token=settings.hf_token or None, sandbox_root=settings.data_dir
    )
    for k, v in results.items():
        console.print(f"  [green]{k}[/] → {v}")
