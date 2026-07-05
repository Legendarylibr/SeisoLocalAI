"""Terminal command validation and subprocess environment scrubbing."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from forge.security.audit import audit_event
from forge.security.code_policy import assert_terminal_allowed, classify_terminal_argv
from seiso.security import SecurityError

_SHELL_METACHAR = re.compile(r"[;|&<>$`\\()\n\r]")

_BLOCKED_EXECUTABLES = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "dash",
        "fish",
        "ksh",
        "csh",
        "tcsh",
        "env",
        "printenv",
        "export",
        "eval",
        "exec",
        "source",
        ".",
        "curl",
        "wget",
        "nc",
        "netcat",
        "ncat",
        "ssh",
        "scp",
        "sftp",
        "telnet",
        "openssl",
        "nmap",
        "socat",
    }
)

_ALLOWED_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "TERM",
        "PWD",
        "SHELL",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "NODE_PATH",
        "GOPATH",
        "GOROOT",
        "RUSTUP_HOME",
        "CARGO_HOME",
        "PYENV_ROOT",
        "NVM_DIR",
    }
)

_SENSITIVE_ENV_PREFIXES = (
    "SEISO_",
    "HF_",
    "OPENAI_",
    "AWS_",
    "AZURE_",
    "GCP_",
    "GOOGLE_",
    "ANTHROPIC_",
    "DATABASE_",
    "REDIS_",
    "SECRET_",
    "TOKEN_",
    "API_KEY",
)

_SENSITIVE_ENV_KEYS = frozenset(
    {
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "NPM_TOKEN",
        "API_KEY",
        "SECRET_KEY",
        "DATABASE_URL",
        "REDIS_URL",
        "INFERENCE_API_KEY",
    }
)


def validate_terminal_command(command: str, *, destructive_ack: bool = False) -> list[str]:
    """Parse a single argv vector; reject shell metacharacters and blocked binaries."""
    command = command.strip()
    if not command:
        raise ValueError("Command is required")
    if _SHELL_METACHAR.search(command):
        raise SecurityError("Shell metacharacters are not allowed in commands")
    try:
        argv = shlex.split(command, posix=os.name != "nt")
    except ValueError as exc:
        raise ValueError(f"Invalid command syntax: {exc}") from exc
    if not argv:
        raise ValueError("Command is required")

    executable = Path(argv[0]).name.lower()
    if executable in _BLOCKED_EXECUTABLES:
        raise SecurityError(f"Command not allowed: {argv[0]}")

    if executable in {"python", "python3", "py"} and "-c" in argv[1:]:
        raise SecurityError("python -c is not allowed")

    assert_terminal_allowed(argv, destructive_ack=destructive_ack)
    return argv


def terminal_command_tier(command: str) -> str:
    """Return safe|risky|blocked for UI previews."""
    try:
        argv = shlex.split(command.strip(), posix=os.name != "nt")
    except ValueError:
        return "blocked"
    if _SHELL_METACHAR.search(command):
        return "blocked"
    return classify_terminal_argv(argv)


def scrubbed_subprocess_env() -> dict[str, str]:
    """Return a minimal environment without Seiso or provider secrets."""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _ALLOWED_ENV_KEYS:
            env[key] = value
            continue
        upper = key.upper()
        if upper in _SENSITIVE_ENV_KEYS:
            continue
        if any(upper.startswith(prefix) for prefix in _SENSITIVE_ENV_PREFIXES):
            continue
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("LANG", os.environ.get("LANG", "C.UTF-8"))
    if "PATH" not in env and "PATH" in os.environ:
        env["PATH"] = os.environ["PATH"]
    if "HOME" not in env and "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    return env


def audit_terminal_command(command: str, *, user_id: str | None = None) -> None:
    """Log terminal invocation without storing full command text."""
    import hashlib

    digest = hashlib.sha256(command.encode()).hexdigest()[:16]
    audit_event(
        "terminal_exec",
        user_id=user_id,
        command_len=len(command),
        command_hash=digest,
    )
