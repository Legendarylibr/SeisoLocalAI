from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

from adaptive_quant.ui.browser import open_dashboard_url
from adaptive_quant.ui.security import validate_bind_host
from adaptive_quant.ui.server import serve_launcher


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_python(repo: Path) -> str:
    if os.name == "nt":
        candidate = repo / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo / ".venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def _open_browser(url: str, *, delay_s: float = 0.6) -> None:
    open_dashboard_url(url, delay_s=delay_s)


def resolve_launcher_port(port: int | None = None) -> int:
    if port is not None:
        return port
    raw = os.environ.get("ADAPTIVE_RL_LAUNCHER_PORT", "8765").strip()
    try:
        return int(raw)
    except ValueError:
        return 8765


def launcher_dashboard_url(*, port: int | None = None, host: str = "127.0.0.1") -> str:
    return f"http://{host}:{resolve_launcher_port(port)}/"


def launcher_autostart_suppressed() -> bool:
    if os.environ.get("ADAPTIVE_RL_SUPPRESS_LAUNCHER_HINT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    return os.environ.get("CI", "").lower() in {"1", "true", "yes"}


def ensure_launcher_running(
    *,
    port: int | None = None,
    repo: Path | None = None,
    python_bin: str | None = None,
) -> str | None:
    """Start the detached launcher when it is not already responding; return its URL."""
    if launcher_autostart_suppressed():
        return None
    from adaptive_quant.ui.browser import wait_for_launcher_ready

    resolved_port = resolve_launcher_port(port)
    url = launcher_dashboard_url(port=resolved_port)
    if wait_for_launcher_ready(url, timeout_s=1.0):
        return url

    resolved_repo = (repo or _repo_root()).resolve()
    resolved_python = python_bin or _default_python(resolved_repo)
    spawn_detached_process(
        [
            resolved_python,
            "-m",
            "adaptive_quant.ui.launcher",
            "--detach",
            "--no-browser",
            "--port",
            str(resolved_port),
            "--repo",
            str(resolved_repo),
            "--python",
            resolved_python,
        ],
        cwd=resolved_repo,
    )
    if not wait_for_launcher_ready(url, timeout_s=15.0):
        print(
            f"Warning: launcher did not respond at {url}. "
            "Start manually: make ui  (or: adaptive-rl-quant-ui)",
            file=sys.stderr,
        )
    return url


def spawn_detached_process(cmd: list[str], *, cwd: Path) -> subprocess.Popen[bytes]:
    kwargs: dict[str, object] = {
        "cwd": str(cwd),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS (Windows-only; not in subprocess on Unix).
        kwargs["creationflags"] = 0x00000200 | 0x00000008
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)  # type: ignore[call-overload]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adaptive RL Quantization launcher dashboard (local web UI)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765).")
    parser.add_argument("--repo", type=Path, default=None, help="Repository root (auto-detected).")
    parser.add_argument(
        "--python", dest="python_bin", default=None, help="Python for subprocess jobs."
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab.")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Spawn a background server process and exit (used by setup).",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding to non-loopback addresses (sets ADAPTIVE_RL_LAUNCHER_ALLOW_REMOTE=1).",
    )
    args = parser.parse_args(argv)

    if args.allow_remote:
        os.environ["ADAPTIVE_RL_LAUNCHER_ALLOW_REMOTE"] = "1"
    validate_bind_host(args.host)

    repo = (args.repo or _repo_root()).resolve()
    python_bin = args.python_bin or _default_python(repo)
    url = f"http://{args.host}:{args.port}/"

    if args.detach:
        cmd = [
            python_bin,
            "-m",
            "adaptive_quant.ui.launcher",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--repo",
            str(repo),
            "--python",
            python_bin,
        ]
        if args.no_browser:
            cmd.append("--no-browser")
        if args.allow_remote:
            cmd.append("--allow-remote")
        spawn_detached_process(cmd, cwd=repo)
        print(f"Launcher UI starting at {url}")
        return 0

    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    print(f"Launcher UI: {url}")
    print("Press Ctrl+C to stop.")
    try:
        serve_launcher(
            repo=repo,
            python_bin=python_bin,
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
