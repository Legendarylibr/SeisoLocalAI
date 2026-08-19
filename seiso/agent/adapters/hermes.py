"""Hermes Agent oneshot adapter (``hermes -z``)."""

from __future__ import annotations

from pathlib import Path

from seiso.agent.adapters.base import BaseAdapter
from seiso.agent.adapters.types import LaunchSpec


class HermesAdapter(BaseAdapter):
    id = "hermes"

    def argv(self, spec: LaunchSpec, binary: str) -> list[str]:
        args = [binary, "-z", spec.goal]
        if spec.model_id and spec.model_id not in {"default", "auto"}:
            args.extend(["--model", spec.model_id])
        return args

    def child_env(self, spec: LaunchSpec) -> dict[str, str]:
        env = super().child_env(spec)
        env["HERMES_INFERENCE_MODEL"] = spec.model_id or "default"
        if spec.endpoint_url:
            env["HERMES_BASE_URL"] = spec.endpoint_url
        # Isolated overlay — do not point HERMES_HOME at ~/.hermes.
        env["HERMES_CONFIG"] = str(Path(spec.isolated_config_dir) / "config.yaml")
        return env
