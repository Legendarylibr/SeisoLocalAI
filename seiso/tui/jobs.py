"""Run documented CLI jobs from the TUI studio pages."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_cli_job(spec: str, *, cwd: Path) -> str:
    raw = spec.strip()
    if not raw:
        return "Usage: /run configs/example_lora.yaml   or   /run train configs/example_lora.yaml"
    parts = raw.split()
    if parts[0] in {"train", "compress", "distill-rl", "export"}:
        cmd = ["seiso", *parts]
        if parts[0] == "compress" and "run" not in parts:
            cmd = ["seiso", "compress", "run", *parts[1:]]
        if parts[0] == "distill-rl" and "run" not in parts:
            cmd = ["seiso", "distill-rl", "run", *parts[1:]]
    else:
        cmd = ["seiso", "train", "--config", parts[0]]
    try:
        subprocess.Popen(cmd, cwd=str(cwd), start_new_session=True)  # noqa: S603
    except FileNotFoundError:
        return "seiso CLI not on PATH. Activate the venv."
    except OSError as exc:
        return f"Could not start `{' '.join(cmd)}`: {exc}"
    return f"Started `{' '.join(cmd)}` in the background. Artifacts land under outputs/ or ~/.seiso/."
