"""OpenClaw (formerly ClawdBot) headless adapter."""

from __future__ import annotations

from seiso.agent.adapters.base import BaseAdapter
from seiso.agent.adapters.types import LaunchSpec


class OpenClawAdapter(BaseAdapter):
    id = "openclaw"

    def argv(self, spec: LaunchSpec, binary: str) -> list[str]:
        return [binary, "agent", "--message", spec.goal, "--json"]

    def child_env(self, spec: LaunchSpec) -> dict[str, str]:
        env = super().child_env(spec)
        # Isolated state so we never rewrite ~/.openclaw.
        env["OPENCLAW_HOME"] = spec.isolated_config_dir
        env["OPENCLAW_STATE_DIR"] = spec.isolated_config_dir
        return env
