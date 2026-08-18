"""Shared oneshot launch + adapter protocol."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from seiso.agent.adapters.detect import detect_harness
from seiso.agent.adapters.endpoint import ResolvedEndpoint
from seiso.agent.adapters.profiles import write_profile
from seiso.agent.adapters.types import DetectedHarness, LaunchResult, LaunchSpec


class HarnessAdapter(Protocol):
    id: str

    def detect(self) -> DetectedHarness: ...

    def configure(self, endpoint: ResolvedEndpoint, isolated_config_dir: Path) -> Path: ...

    def launch(self, spec: LaunchSpec) -> LaunchResult: ...


def run_oneshot(
    harness_id: str,
    argv: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str] | None = None,
    timeout_sec: int = 600,
) -> LaunchResult:
    if not argv:
        return LaunchResult(harness_id, 127, detail="empty_argv")
    merged = os.environ.copy()
    if env:
        merged.update({k: v for k, v in env.items() if v is not None})
    try:
        proc = subprocess.run(  # noqa: S603
            list(argv),
            cwd=cwd or None,
            env=merged,
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout_sec)),
            check=False,
        )
    except FileNotFoundError:
        return LaunchResult(harness_id, 127, detail=f"missing_binary:{argv[0]}")
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return LaunchResult(
            harness_id,
            124,
            stdout=(out or "")[-4000:],
            stderr=(err or "")[-2000:],
            detail="timeout",
        )
    except OSError as exc:
        return LaunchResult(harness_id, 1, detail=f"spawn_failed:{exc}")
    return LaunchResult(
        harness_id,
        int(proc.returncode),
        stdout=(proc.stdout or "")[-8000:],
        stderr=(proc.stderr or "")[-4000:],
        detail="ok" if proc.returncode == 0 else f"exit:{proc.returncode}",
    )


class BaseAdapter:
    id = "pi"

    def detect(self) -> DetectedHarness:
        return detect_harness(self.id)

    def configure(self, endpoint: ResolvedEndpoint, isolated_config_dir: Path) -> Path:
        return write_profile(isolated_config_dir, self.id, endpoint)

    def argv(self, spec: LaunchSpec, binary: str) -> list[str]:
        return [binary, "--help"]

    def child_env(self, spec: LaunchSpec) -> dict[str, str]:
        env = {
            "OPENAI_BASE_URL": spec.endpoint_url,
            "OPENAI_API_KEY": spec.api_key or "not-needed",
            "OPENAI_MODEL": spec.model_id,
        }
        env.update(dict(spec.extra_env))
        return env

    def launch(self, spec: LaunchSpec) -> LaunchResult:
        found = self.detect()
        if not found.installed or not found.binary:
            return LaunchResult(
                self.id,
                127,
                detail=f"not_installed:{self.id}",
            )
        return run_oneshot(
            self.id,
            self.argv(spec, found.binary),
            cwd=spec.workdir,
            env=self.child_env(spec),
            timeout_sec=spec.timeout_sec,
        )
