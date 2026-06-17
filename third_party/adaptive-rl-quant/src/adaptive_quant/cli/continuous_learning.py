"""CLI: continuous RL over a streaming task sequence."""

from __future__ import annotations

import argparse

from adaptive_quant.cli.common import (
    add_config_file_argument,
    add_config_override_arguments,
    load_config_or_fallback,
    resolve_startup_config,
)
from adaptive_quant.continuous_pipeline import run_continuous_pipeline_entrypoint
from adaptive_quant.presets.continuous import CONFIG_CONTINUOUS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuous learning pipeline: RL policy trains on a streaming task sequence "
            "with replay-buffer updates, periodic eval, and drift rollback."
        )
    )
    add_config_file_argument(parser)
    add_config_override_arguments(parser)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Override the number of continuous tasks (defaults to config.continuous_max_tasks).",
    )
    args = parser.parse_args()
    from adaptive_quant.cli.common import enforce_cli_startup

    enforce_cli_startup(context="continuous learning CLI")
    cfg, cli_overrides = resolve_startup_config(
        load_config_or_fallback(args.config, CONFIG_CONTINUOUS),
        args,
    )
    run_continuous_pipeline_entrypoint(
        cfg,
        max_tasks=args.max_tasks,
        cli_startup_overrides=cli_overrides,
    )


if __name__ == "__main__":
    main()
