from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser


def format_terminal_hyperlink(url: str, *, label: str | None = None) -> str:
    """Return a terminal hyperlink (OSC 8) when stdout is an interactive TTY."""
    text = label or url
    if os.environ.get("CI", "").lower() in {"1", "true", "yes"}:
        return url
    if os.environ.get("NO_COLOR", "").strip():
        return url
    if not sys.stdout.isatty():
        return url
    if os.environ.get("TERM", "").lower() == "dumb":
        return url
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def wait_for_launcher_ready(url: str, *, timeout_s: float = 15.0, poll_s: float = 0.2) -> bool:
    """Poll the launcher URL until it responds or the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=min(1.0, timeout_s)) as response:
                if 200 <= response.status < 400:
                    return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def open_dashboard_url(
    url: str,
    *,
    delay_s: float = 1.0,
    wait_for_ready: bool = False,
    ready_timeout_s: float = 15.0,
) -> bool:
    """Open the launcher dashboard in the system browser (Linux, macOS, Windows)."""
    if os.environ.get("CI", "").lower() in {"1", "true", "yes"}:
        return False
    if wait_for_ready:
        wait_for_launcher_ready(url, timeout_s=ready_timeout_s)
    elif delay_s > 0:
        time.sleep(delay_s)
    try:
        if webbrowser.open(url, new=2):
            return True
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Darwin":
            completed = subprocess.run(["open", url], check=False, capture_output=True)
            return completed.returncode == 0
        if system == "Windows":
            completed = subprocess.run(
                ["cmd", "/c", "start", "", url],
                check=False,
                capture_output=True,
            )
            return completed.returncode == 0
        xdg_open = shutil.which("xdg-open")
        if xdg_open:
            completed = subprocess.run([xdg_open, url], check=False, capture_output=True)
            return completed.returncode == 0
    except OSError:
        return False
    return False
