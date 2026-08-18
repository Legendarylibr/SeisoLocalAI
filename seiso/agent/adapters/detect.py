"""Discover optional harness CLIs. PATH + well-known homes only — no network."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from seiso.agent.adapters.types import HARNESS_IDS, HARNESS_LABELS, DetectedHarness

_BINARIES: dict[str, tuple[str, ...]] = {
    "pi": ("pi",),
    "omp": ("omp",),
    "hermes": ("hermes",),
    "cline": ("cline",),
    "openclaw": ("openclaw", "clawdbot"),
}

_HOMES: dict[str, tuple[str, ...]] = {
    "pi": (".pi",),
    "omp": (".omp",),
    "hermes": (".hermes",),
    "cline": (".cline",),
    "openclaw": (".openclaw", ".clawdbot"),
}

_HINTS: dict[str, str] = {
    "pi": "npm i -g @mariozechner/pi-coding-agent",
    "omp": "see https://omp.sh",
    "hermes": "see https://hermes-agent.nousresearch.com",
    "cline": "npm i -g cline",
    "openclaw": "see https://docs.openclaw.ai",
}


def _home_dir(harness_id: str) -> Path | None:
    home = Path.home()
    for name in _HOMES.get(harness_id, ()):
        path = home / name
        if path.is_dir():
            return path
    return None


def _version(binary: str) -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (proc.stdout or proc.stderr or "").strip().splitlines()
    if not text:
        return None
    return text[0][:80]


def detect_harness(harness_id: str, *, include_version: bool = False) -> DetectedHarness:
    label = HARNESS_LABELS.get(harness_id, harness_id)
    binary: str | None = None
    for name in _BINARIES.get(harness_id, (harness_id,)):
        found = shutil.which(name)
        if found:
            binary = found
            break
    home = _home_dir(harness_id)
    installed = binary is not None
    return DetectedHarness(
        id=harness_id,
        label=label,
        installed=installed,
        binary=binary,
        version=_version(binary) if binary and include_version else None,
        home=str(home) if home is not None else None,
        hint="" if installed else _HINTS.get(harness_id, ""),
    )


def detect_all(*, include_version: bool = False) -> tuple[DetectedHarness, ...]:
    return tuple(detect_harness(hid, include_version=include_version) for hid in HARNESS_IDS)


def default_harness_id(detected: tuple[DetectedHarness, ...] | None = None) -> str:
    rows = detected if detected is not None else detect_all()
    for preferred in ("hermes", "pi", "omp", "openclaw", "cline"):
        row = next((item for item in rows if item.id == preferred and item.installed), None)
        if row is not None:
            return row.id
    env = (os.environ.get("SEISO_AGENT_HARNESS") or "").strip().lower()
    if env in HARNESS_IDS:
        return env
    return "hermes"
