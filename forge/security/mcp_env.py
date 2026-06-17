"""Immutable MCP subprocess environment — block PATH and injection keys."""

from __future__ import annotations

_BLOCKED_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
        "NODE_OPTIONS",
        "NODE_CHANNEL_FD",
        "NODE_ICU_DATA",
        "BASH_ENV",
        "ENV",
        "GCONV_PATH",
        "IFS",
        "SSLKEYLOGFILE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "FTP_PROXY",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    }
)

_BLOCKED_PREFIXES = ("LD_", "DYLD_", "NODE_", "PYTHON", "SSL_")


def is_blocked_env_key(key: str) -> bool:
    upper = key.upper()
    if upper in _BLOCKED_EXACT:
        return True
    return any(upper.startswith(p) for p in _BLOCKED_PREFIXES)


def sanitize_mcp_env(env: dict[str, str]) -> dict[str, str]:
    """Drop dangerous keys from user-supplied MCP environment."""
    return {k: v for k, v in env.items() if not is_blocked_env_key(k)}


def mcp_subprocess_env(user_env: dict[str, str]) -> dict[str, str]:
    """Build fixed subprocess env; user keys cannot override base entries."""
    base = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    extra = sanitize_mcp_env(user_env)
    for key in base:
        extra.pop(key, None)
    return {**base, **extra}
