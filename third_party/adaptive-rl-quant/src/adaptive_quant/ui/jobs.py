from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_quant.ui.commands import build_job_env, build_workflow_command, format_command
from adaptive_quant.ui.security import validate_run_options


@dataclass
class JobRecord:
    job_id: str
    workflow: str
    label: str
    command: list[str]
    cwd: str
    status: str = "queued"
    exit_code: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    log_lines: list[str] = field(default_factory=list)
    _process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "workflow": self.workflow,
            "action": self.workflow,
            "label": self.label,
            "command": self.command,
            "command_preview": format_command(self.command),
            "cwd": self.cwd,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_tail": self.log_lines[-200:],
        }


class JobManager:
    def __init__(self, *, max_jobs: int = 20) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def _trim(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        finished = sorted(
            (job for job in self._jobs.values() if job.status in {"succeeded", "failed"}),
            key=lambda item: item.finished_at or 0.0,
        )
        for job in finished[: max(0, len(self._jobs) - self._max_jobs)]:
            self._jobs.pop(job.job_id, None)

    def start(
        self,
        *,
        workflow: str,
        label: str,
        command: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            job_id=job_id,
            workflow=workflow,
            label=label,
            command=command,
            cwd=str(cwd),
        )
        thread = threading.Thread(
            target=self._run_job,
            args=(record, env or dict(os.environ)),
            daemon=True,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._trim()
        thread.start()
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                job.to_dict()
                for job in sorted(self._jobs.values(), key=lambda j: j.started_at or 0.0)
            ]

    def _append_log(self, record: JobRecord, line: str) -> None:
        with self._lock:
            record.log_lines.append(line.rstrip("\n"))
            if len(record.log_lines) > 2000:
                record.log_lines = record.log_lines[-2000:]

    def _run_job(self, record: JobRecord, env: dict[str, str]) -> None:
        record.status = "running"
        record.started_at = time.time()
        try:
            process = subprocess.Popen(
                record.command,
                cwd=record.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            record._process = process
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(record, line)
            exit_code = process.wait()
            record.exit_code = exit_code
            record.status = "succeeded" if exit_code == 0 else "failed"
        except OSError as exc:
            self._append_log(record, f"ERROR: {exc}")
            record.exit_code = 1
            record.status = "failed"
        finally:
            record.finished_at = time.time()
            record._process = None


def build_action_command(*, action: str, repo: Path, python_bin: str) -> tuple[str, list[str]]:
    """Backward-compatible wrapper around :func:`build_workflow_command`."""
    return build_workflow_command(
        workflow=action,
        options={},
        repo=repo,
        python_bin=python_bin,
    )


def start_configured_workflow(
    *,
    jobs: JobManager,
    repo: Path,
    python_bin: str,
    workflow: str,
    options: dict[str, Any] | None = None,
) -> JobRecord:
    opts = dict(options or {})
    validate_run_options(opts)
    label, command = build_workflow_command(
        workflow=workflow,
        options=opts,
        repo=repo,
        python_bin=python_bin,
    )
    env = build_job_env(opts, repo=repo)
    return jobs.start(workflow=workflow, label=label, command=command, cwd=repo, env=env)


def action_catalog(status: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact quick-action tiles derived from the workflow catalog."""
    setup = status.get("setup", {})
    torch_info = status.get("torch", {})
    nvidia = status.get("nvidia", {})
    ready = bool(setup.get("package_importable"))
    cuda_ready = bool(torch_info.get("cuda_available"))
    needs_ack = bool(nvidia.get("needs_ack_for_gpu_training"))

    quick: list[dict[str, Any]] = [
        {
            "id": "smoke",
            "workflow": "research",
            "title": "Quick smoke test",
            "description": "config.e2e_smoke.json end-to-end run.",
            "category": "run",
            "enabled": ready,
            "primary": True,
            "options": {"config": "config.e2e_smoke.json"},
        },
        {
            "id": "full_run",
            "workflow": "research",
            "title": "Full simulator run",
            "description": "Baseline research pipeline.",
            "category": "run",
            "enabled": ready,
            "primary": True,
            "options": {},
        },
        {
            "id": "cuda_check",
            "workflow": "cuda_check",
            "title": "CUDA diagnostics",
            "description": "Check torch/CUDA without installing.",
            "category": "verify",
            "enabled": ready,
        },
        {
            "id": "setup_tests",
            "workflow": "setup_tests",
            "title": "Setup tests",
            "description": "Re-run hardware-aware setup tests.",
            "category": "verify",
            "enabled": ready,
            "options": {},
        },
        {
            "id": "doctor",
            "workflow": "doctor",
            "title": "Environment report",
            "description": "Detailed doctor output.",
            "category": "verify",
            "enabled": ready,
            "options": {},
        },
    ]

    if nvidia.get("linux_nvidia_host"):
        quick.append(
            {
                "id": "install_cuda",
                "workflow": "install_cuda",
                "title": "Install CUDA PyTorch",
                "description": "GPU torch wheel (cu130 default).",
                "category": "gpu",
                "enabled": ready and not cuda_ready,
                "primary": not cuda_ready,
                "options": {"accept_gpu_install": True, "nvidia_ack": "host_venv"},
                "requires_ack": needs_ack,
            }
        )
        for preset, title in (
            ("gpu", "PyTorch GPU"),
            ("4090", "PyTorch 4090"),
            ("3090", "PyTorch 3090"),
        ):
            quick.append(
                {
                    "id": f"pytorch:{preset}",
                    "workflow": "pytorch",
                    "title": title,
                    "description": f"CUDA trainer preset={preset}.",
                    "category": "gpu",
                    "enabled": ready and cuda_ready,
                    "options": {"preset": preset, "nvidia_ack": "host_venv"},
                    "requires_ack": needs_ack,
                }
            )
    elif ready and torch_info.get("torch_installed"):
        quick.append(
            {
                "id": "pytorch:gpu",
                "workflow": "pytorch",
                "title": "PyTorch run",
                "description": "CPU/CUDA if available.",
                "category": "gpu",
                "enabled": True,
                "options": {"preset": "gpu"},
            }
        )

    return quick
