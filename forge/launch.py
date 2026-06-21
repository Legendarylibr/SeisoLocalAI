"""Forge launch helpers — health wait and browser open."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import webbrowser


def wait_for_health(url: str, *, timeout_s: float = 30.0, poll_s: float = 0.5) -> bool:
    """Poll /health until Forge responds or timeout."""
    health_url = f"{url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=min(2.0, timeout_s)) as response:
                if 200 <= response.status < 400:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(poll_s)
    return False


def open_browser(url: str) -> bool:
    """Open URL in the system browser."""
    if os.environ.get("SEISO_NO_OPEN", "").strip() in {"1", "true", "yes"}:
        return False
    if os.environ.get("CI", "").lower() in {"1", "true", "yes"}:
        return False

    try:
        if webbrowser.open(url, new=2):
            return True
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.run(["open", url], check=False, capture_output=True).returncode == 0
        if system == "Windows":
            return (
                subprocess.run(
                    ["cmd", "/c", "start", "", url], check=False, capture_output=True
                ).returncode
                == 0
            )
        xdg_open = shutil.which("xdg-open")
        if xdg_open:
            return subprocess.run([xdg_open, url], check=False, capture_output=True).returncode == 0
    except OSError:
        return False
    return False


def open_forge_when_ready(url: str, *, timeout_s: float = 30.0) -> None:
    """Background-friendly: wait for /health, then open the dashboard."""
    if wait_for_health(url, timeout_s=timeout_s):
        open_browser(url)


def schedule_browser_open(url: str, *, timeout_s: float = 30.0) -> None:
    """Start a daemon thread that opens Forge when the server is ready."""
    thread = threading.Thread(
        target=open_forge_when_ready,
        args=(url,),
        kwargs={"timeout_s": timeout_s},
        daemon=True,
        name="seiso-open-browser",
    )
    thread.start()
