"""Cline CLI headless adapter."""

from __future__ import annotations

from seiso.agent.adapters.base import BaseAdapter
from seiso.agent.adapters.types import LaunchSpec


class ClineAdapter(BaseAdapter):
    id = "cline"

    def argv(self, spec: LaunchSpec, binary: str) -> list[str]:
        return [binary, "chat", "--print", spec.goal]

    def child_env(self, spec: LaunchSpec) -> dict[str, str]:
        env = super().child_env(spec)
        env["CLINE_DIR"] = spec.isolated_config_dir
        return env
