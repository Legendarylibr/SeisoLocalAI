"""Types for optional external coding-agent harness adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

HARNESS_IDS: tuple[str, ...] = ("pi", "omp", "hermes", "cline", "openclaw")
HARNESS_LABELS: dict[str, str] = {
    "pi": "Pi",
    "omp": "OMP",
    "hermes": "Hermes",
    "cline": "Cline",
    "openclaw": "OpenClaw",
}


def parse_harness_id(raw: str | None) -> str:
    text = str(raw or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {"ohmypi": "omp", "oh-my-pi": "omp", "clawdbot": "openclaw", "clawd": "openclaw"}
    text = aliases.get(text, text)
    if text not in HARNESS_IDS:
        allowed = ", ".join(HARNESS_IDS)
        raise ValueError(f"unknown harness {raw!r}; expected one of: {allowed}")
    return text


@dataclass(frozen=True, slots=True)
class DetectedHarness:
    id: str
    label: str
    installed: bool
    binary: str | None = None
    version: str | None = None
    home: str | None = None
    hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    goal: str
    workdir: str
    isolated_config_dir: str
    endpoint_url: str
    model_id: str
    api_key: str = ""
    timeout_sec: int = 600
    extra_env: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extra_env"] = dict(self.extra_env)
        data.pop("api_key", None)
        return data


@dataclass(frozen=True, slots=True)
class LaunchResult:
    harness_id: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    detail: str = ""
    artifacts: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "detail": self.detail,
            "artifacts": list(self.artifacts),
            "ok": self.ok,
        }
