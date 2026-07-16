"""Optional managed multi-GPU vLLM process lifecycle.

Opt-in only. Default Seiso start does not launch vLLM. Enable via Forge API,
Integrations UI, or ``SEISO_MANAGED_VLLM_AUTOSTART=1`` with a model set.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from seiso.env import env_bool, env_int, env_str
from seiso.hardware.gpus import gpu_count

logger = logging.getLogger(__name__)

_MANAGED_LOCK = threading.RLock()
_STATE: ManagedVllmState | None = None

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_HEALTH_PATH = "/v1/models"
DEFAULT_START_TIMEOUT_S = 600
MANAGED_PROVIDER_MARKER = "seiso_managed_vllm"


def managed_vllm_enabled() -> bool:
    """Feature gate — off by default (optional path)."""
    return env_bool("SEISO_MANAGED_VLLM_ENABLED", False) or env_bool(
        "SEISO_ALLOW_MANAGED_VLLM", False
    )


def managed_vllm_autostart_enabled() -> bool:
    """Optional start-path gate — never default-on."""
    return managed_vllm_enabled() and env_bool("SEISO_MANAGED_VLLM_AUTOSTART", False)


def suggest_tensor_parallel(gpu_n: int | None = None) -> int:
    """Pick a vLLM-friendly TP size (power of two, ≤ GPU count)."""
    n = int(gpu_n if gpu_n is not None else gpu_count(include_mlx=False))
    if n <= 1:
        return 1
    tp = 1
    while tp * 2 <= n:
        tp *= 2
    return tp


def resolve_vllm_command() -> list[str] | None:
    """Return argv prefix to invoke a vLLM chat-completions server, or None."""
    override = env_str("SEISO_MANAGED_VLLM_BIN", "")
    if override:
        return [override]
    # Prefer module entry (works in project venv without a `vllm` console script).
    try:
        import vllm  # noqa: F401

        return [os.environ.get("SEISO_PYTHON", "python3"), "-m", "vllm.entrypoints.openai.api_server"]
    except ImportError:
        pass
    from shutil import which

    binary = which("vllm")
    if binary:
        return [binary, "serve"]
    return None


def build_launch_command(
    *,
    model: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    extra_args: list[str] | None = None,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    """Build the managed launch command (preview or execute)."""
    base = resolve_vllm_command()
    if not base:
        raise RuntimeError(
            "vLLM is not installed. Install with: pip install vllm "
            "(or set SEISO_MANAGED_VLLM_BIN to a server binary)."
        )
    model = (model or "").strip()
    if not model:
        raise ValueError("model is required")

    tp = int(tensor_parallel_size or suggest_tensor_parallel())
    if tp < 1:
        raise ValueError("tensor_parallel_size must be >= 1")

    host = (host or DEFAULT_HOST).strip() or DEFAULT_HOST
    port = int(port or DEFAULT_PORT)
    if port < 1 or port > 65535:
        raise ValueError("port out of range")

    # `vllm serve MODEL ...` vs `python -m vllm.entrypoints.openai.api_server --model MODEL`
    cmd = list(base)
    if cmd[-1] == "serve":
        cmd.append(model)
    else:
        cmd.extend(["--model", model])

    cmd.extend(
        [
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tp),
        ]
    )
    util = gpu_memory_utilization
    if util is None:
        util_raw = env_str("SEISO_MANAGED_VLLM_GPU_MEMORY_UTILIZATION", "")
        util = float(util_raw) if util_raw else None
    if util is not None:
        if not 0.1 <= float(util) <= 1.0:
            raise ValueError("gpu_memory_utilization must be between 0.1 and 1.0")
        cmd.extend(["--gpu-memory-utilization", str(float(util))])

    mml = max_model_len if max_model_len is not None else env_int("SEISO_MANAGED_VLLM_MAX_MODEL_LEN", 0)
    if mml and mml > 0:
        cmd.extend(["--max-model-len", str(int(mml))])

    if extra_args:
        cmd.extend(str(a) for a in extra_args if str(a).strip())

    # Optional LoRA hot-reload for multi-GPU slime RL (dynamic /v1/load_lora_adapter).
    enable_lora = env_bool("SEISO_MANAGED_VLLM_ENABLE_LORA", False)
    if enable_lora and "--enable-lora" not in cmd:
        cmd.append("--enable-lora")

    env: dict[str, str] = {}
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
    return {
        "command": cmd,
        "env": env,
        "host": host,
        "port": port,
        "model": model,
        "tensor_parallel_size": tp,
        "base_url": f"http://{host}:{port}/v1",
        "gpu_count": gpu_count(include_mlx=False),
    }


@dataclass
class ManagedVllmState:
    pid: int
    command: list[str]
    host: str
    port: int
    model: str
    tensor_parallel_size: int
    base_url: str
    log_path: str
    started_at: float
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    managed: bool = True

    def to_status(self) -> dict[str, Any]:
        alive = self.is_alive()
        return {
            "running": alive,
            "managed": True,
            "pid": self.pid if alive else None,
            "host": self.host,
            "port": self.port,
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "base_url": self.base_url,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "healthy": alive and _health_ok(self.base_url),
            "enabled": managed_vllm_enabled(),
            "autostart": managed_vllm_autostart_enabled(),
        }

    def is_alive(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False


def get_status() -> dict[str, Any]:
    with _MANAGED_LOCK:
        if _STATE is None:
            return {
                "running": False,
                "managed": False,
                "enabled": managed_vllm_enabled(),
                "autostart": managed_vllm_autostart_enabled(),
                "vllm_available": resolve_vllm_command() is not None,
                "suggested_tensor_parallel": suggest_tensor_parallel(),
                "gpu_count": gpu_count(include_mlx=False),
            }
        status = _STATE.to_status()
        status["vllm_available"] = resolve_vllm_command() is not None
        status["suggested_tensor_parallel"] = suggest_tensor_parallel()
        status["gpu_count"] = gpu_count(include_mlx=False)
        if not status["running"]:
            _clear_state_unlocked()
            status["running"] = False
            status["healthy"] = False
            status["pid"] = None
        return status


def _pid_file(data_dir: Path) -> Path:
    return Path(data_dir) / "run" / "managed-vllm.pid"


def _log_file(data_dir: Path) -> Path:
    return Path(data_dir) / "run" / "managed-vllm.log"


def _port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _health_ok(base_url: str, timeout: float = 2.0) -> bool:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        # Accept either .../v1 or bare host:port
        health = f"{url}/v1/models" if not url.endswith("/models") else url
    else:
        health = f"{url}/models"
    try:
        req = Request(health, method="GET")
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — localhost managed endpoint
            return 200 <= int(getattr(resp, "status", 200)) < 300
    except (URLError, TimeoutError, OSError, ValueError):
        return False


def _clear_state_unlocked() -> None:
    global _STATE
    _STATE = None


def start_managed_vllm(
    *,
    model: str,
    data_dir: Path,
    host: str | None = None,
    port: int | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    cuda_visible_devices: str | None = None,
    extra_args: list[str] | None = None,
    wait_ready: bool = True,
    start_timeout_s: int | None = None,
) -> dict[str, Any]:
    """Start a managed multi-GPU vLLM chat server (local loopback only)."""
    if not managed_vllm_enabled():
        raise RuntimeError(
            "Managed multi-GPU vLLM is disabled. Set SEISO_MANAGED_VLLM_ENABLED=true "
            "to use this optional path."
        )

    host = (host or env_str("SEISO_MANAGED_VLLM_HOST", DEFAULT_HOST)).strip() or DEFAULT_HOST
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "Managed vLLM must bind to loopback (127.0.0.1). "
            "For remote multi-GPU, configure a cloud multi-GPU provider instead."
        )
    port = int(port or env_int("SEISO_MANAGED_VLLM_PORT", DEFAULT_PORT))
    model = (model or env_str("SEISO_MANAGED_VLLM_MODEL", "")).strip()
    if not model:
        raise ValueError("model is required (or set SEISO_MANAGED_VLLM_MODEL)")

    launch = build_launch_command(
        model=model,
        host=host,
        port=port,
        tensor_parallel_size=tensor_parallel_size
        or env_int("SEISO_MANAGED_VLLM_TENSOR_PARALLEL", 0)
        or None,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        extra_args=extra_args,
        cuda_visible_devices=cuda_visible_devices
        or env_str("SEISO_MANAGED_VLLM_CUDA_VISIBLE_DEVICES", "")
        or None,
    )

    with _MANAGED_LOCK:
        global _STATE
        if _STATE is not None and _STATE.is_alive():
            if (
                _STATE.model == launch["model"]
                and _STATE.port == launch["port"]
                and _STATE.tensor_parallel_size == launch["tensor_parallel_size"]
            ):
                return _STATE.to_status()
            raise RuntimeError(
                "Managed vLLM is already running "
                f"(pid={_STATE.pid}, model={_STATE.model}). Stop it first."
            )

        if _port_in_use(launch["host"], launch["port"]) and not _health_ok(launch["base_url"]):
            raise RuntimeError(
                f"Port {launch['port']} is in use but not a healthy vLLM /v1 endpoint. "
                "Free the port or choose another SEISO_MANAGED_VLLM_PORT."
            )
        if _health_ok(launch["base_url"]):
            # External healthy server on the target port — adopt as unmanaged connect.
            _STATE = ManagedVllmState(
                pid=0,
                command=[],
                host=launch["host"],
                port=launch["port"],
                model=launch["model"],
                tensor_parallel_size=launch["tensor_parallel_size"],
                base_url=launch["base_url"],
                log_path="",
                started_at=time.time(),
                process=None,
                managed=False,
            )
            status = _STATE.to_status()
            status["adopted_existing"] = True
            status["managed"] = False
            return status

        data_dir = Path(data_dir).expanduser()
        run_dir = data_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = _log_file(data_dir)
        pid_path = _pid_file(data_dir)

        env = os.environ.copy()
        env.update(launch["env"])
        # Keep managed serve isolated from Forge's in-process CUDA usage.
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        log_fh = open(log_path, "ab", buffering=0)  # noqa: SIM115
        try:
            popen_kwargs: dict[str, Any] = {
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "env": env,
            }
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(launch["command"], **popen_kwargs)  # noqa: S603
        except Exception:
            log_fh.close()
            raise

        pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
        pid_path.chmod(0o600)
        _STATE = ManagedVllmState(
            pid=proc.pid,
            command=list(launch["command"]),
            host=launch["host"],
            port=launch["port"],
            model=launch["model"],
            tensor_parallel_size=launch["tensor_parallel_size"],
            base_url=launch["base_url"],
            log_path=str(log_path),
            started_at=time.time(),
            process=proc,
            managed=True,
        )
        logger.info(
            "Started managed multi-GPU vLLM pid=%s tp=%s model=%s url=%s",
            proc.pid,
            launch["tensor_parallel_size"],
            launch["model"],
            launch["base_url"],
        )

        if not wait_ready:
            return _STATE.to_status()

        timeout = int(
            start_timeout_s
            if start_timeout_s is not None
            else env_int("SEISO_MANAGED_VLLM_START_TIMEOUT_S", DEFAULT_START_TIMEOUT_S)
        )
        deadline = time.time() + max(30, timeout)
        while time.time() < deadline:
            if proc.poll() is not None:
                _STATE = None
                raise RuntimeError(
                    f"Managed vLLM exited early (code={proc.returncode}). "
                    f"See log: {log_path}"
                )
            if _health_ok(launch["base_url"]):
                return _STATE.to_status()
            time.sleep(1.0)

        stop_managed_vllm(data_dir=data_dir, force=True)
        raise TimeoutError(
            f"Managed vLLM did not become healthy within {timeout}s. See log: {log_path}"
        )


def stop_managed_vllm(*, data_dir: Path | None = None, force: bool = False) -> dict[str, Any]:
    """Stop the managed process if Seiso started it."""
    with _MANAGED_LOCK:
        global _STATE
        state = _STATE
        if state is None:
            # Best-effort cleanup of stale pid file.
            if data_dir is not None:
                pid_path = _pid_file(Path(data_dir))
                if pid_path.exists():
                    try:
                        pid = int(pid_path.read_text(encoding="utf-8").strip())
                        if pid > 0:
                            _terminate_pid(pid, force=force)
                    except (ValueError, OSError):
                        pass
                    try:
                        pid_path.unlink(missing_ok=True)
                    except OSError:
                        logger.debug("Could not remove managed-vllm pid file", exc_info=True)
            return {"stopped": False, "running": False, "reason": "not_running"}

        if not state.managed or state.pid <= 0:
            _clear_state_unlocked()
            return {
                "stopped": False,
                "running": False,
                "reason": "not_managed",
                "message": "Endpoint was adopted, not started by Seiso; stop it externally.",
            }

        _terminate_pid(state.pid, force=force, process=state.process)
        if data_dir is not None:
            try:
                _pid_file(Path(data_dir)).unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove managed-vllm pid file", exc_info=True)
        _clear_state_unlocked()
        logger.info("Stopped managed multi-GPU vLLM (pid=%s)", state.pid)
        return {"stopped": True, "running": False, "pid": state.pid}


def _terminate_pid(
    pid: int,
    *,
    force: bool = False,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    if pid <= 0:
        return
    try:
        if os.name == "posix":
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                if process is not None and process.poll() is None:
                    process.terminate()
            except PermissionError:
                os.kill(pid, signal.SIGTERM)
        elif process is not None:
            process.terminate()
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + (5 if not force else 1)
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.2)

    try:
        if os.name == "posix":
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def maybe_autostart_from_env(*, data_dir: Path) -> dict[str, Any] | None:
    """Optional start-path hook. No-op unless autostart + model are set."""
    if not managed_vllm_autostart_enabled():
        return None
    model = env_str("SEISO_MANAGED_VLLM_MODEL", "")
    if not model:
        logger.warning(
            "SEISO_MANAGED_VLLM_AUTOSTART is set but SEISO_MANAGED_VLLM_MODEL is empty — skipping"
        )
        return None
    try:
        return start_managed_vllm(model=model, data_dir=data_dir, wait_ready=True)
    except Exception:
        logger.exception("Optional managed vLLM autostart failed")
        return None


def provider_config_from_status(status: dict[str, Any]) -> dict[str, Any]:
    """Config blob for a chat provider row pointing at the managed server."""
    return {
        "base_url": status.get("base_url") or f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1",
        "model": status.get("model") or "default",
        "tensor_parallel_size": status.get("tensor_parallel_size") or 1,
        "deployment_kind": "multi_gpu_local",
        "managed_by": MANAGED_PROVIDER_MARKER,
    }
