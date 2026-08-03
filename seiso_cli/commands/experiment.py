"""Experiment sub-app and research benchmark commands."""

from __future__ import annotations

import typer

from seiso_cli.console import console

experiment_app = typer.Typer(
    name="experiment",
    help="Research benchmarks and regression studies.",
    no_args_is_help=True,
)


@experiment_app.callback(invoke_without_command=True)
def experiment_root(ctx: typer.Context) -> None:
    """Adaptive RL quantization research moved to the Adaptive-RL-Quantization repo."""
    if ctx.invoked_subcommand is None:
        console.print(
            "[yellow]No experiment commands are bundled in Seiso.[/] "
            "Adaptive RL quantization research lives in "
            "[link=https://github.com/Legendarylibr/Adaptive-RL-Quantization]"
            "Adaptive-RL-Quantization[/link]."
        )
