"""Pi / OMP (oh-my-pi) headless JSON adapter."""

from __future__ import annotations

from seiso.agent.adapters.base import BaseAdapter
from seiso.agent.adapters.types import LaunchSpec


class PiFamilyAdapter(BaseAdapter):
    """Shared by Pi and OMP — OMP is a Pi fork with the same JSON print mode."""

    id = "pi"

    def argv(self, spec: LaunchSpec, binary: str) -> list[str]:
        return [
            binary,
            "--mode",
            "json",
            "-p",
            spec.goal,
            "--provider",
            "seiso",
        ]

    def child_env(self, spec: LaunchSpec) -> dict[str, str]:
        env = super().child_env(spec)
        env["PI_CODING_AGENT_DIR"] = spec.isolated_config_dir
        env["PI_HEADER"] = "0"
        return env


class PiAdapter(PiFamilyAdapter):
    id = "pi"


class OmpAdapter(PiFamilyAdapter):
    id = "omp"
