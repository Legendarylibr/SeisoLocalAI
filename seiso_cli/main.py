"""Seiso CLI — forge, train, chat, export."""

from __future__ import annotations

import typer

from seiso_cli.bootstrap import bootstrap_runtime
from seiso_cli.commands.chat import bench_inference_cmd, chat, inference_cmd
from seiso_cli.commands.experiment import experiment_app
from seiso_cli.commands.export import export_cmd
from seiso_cli.commands.forge import doctor, forge
from seiso_cli.commands.nemo_rl import nemo_rl
from seiso_cli.commands.pipelines import compress_app, distill_rl_app, rl_quant_app
from seiso_cli.commands.slime import slime
from seiso_cli.commands.train import train

bootstrap_runtime()

app = typer.Typer(
    name="seiso",
    help="Seiso — local AI platform for training, inference, and export.",
    no_args_is_help=True,
)

app.command()(forge)
app.command()(doctor)
app.command()(train)
app.command(name="slime")(slime)
app.command(name="nemo-rl")(nemo_rl)
app.command()(chat)
app.command(name="export")(export_cmd)
app.command(name="inference")(inference_cmd)
app.command(name="bench-inference")(bench_inference_cmd)

app.add_typer(rl_quant_app, name="rl-quant")
app.add_typer(compress_app, name="compress")
app.add_typer(distill_rl_app, name="distill-rl")
app.add_typer(experiment_app, name="experiment")

if __name__ == "__main__":
    app()
