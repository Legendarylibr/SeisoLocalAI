"""Startup security policy validation and warnings."""

from __future__ import annotations

import logging
import os

from forge.config import ForgeSettings

logger = logging.getLogger(__name__)

_REMOTE_ACK_ENV = "SEISO_REMOTE_ACK"
_REMOTE_DANGEROUS_ACK_ENV = "SEISO_REMOTE_DANGEROUS_ACK"
_NVIDIA_HOST_VENV_ACK_ENV = "SEISO_NVIDIA_HOST_VENV_ACK"
_LEGACY_NVIDIA_HOST_VENV_ACK_ENV = "ADAPTIVE_RL_NVIDIA_HOST_VENV_ACK"


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _set_native_linux_nvidia_boundary_for_local_forge(settings: ForgeSettings) -> bool:
    """Approve local native-Linux Forge training jobs for trusted host-venv CUDA."""
    from seiso.security.nvidia_boundary import (
        approved_nvidia_boundary,
        detect_wsl2,
        in_ci,
        is_linux_nvidia_host,
    )

    if (
        settings.allow_remote
        or detect_wsl2()
        or in_ci()
        or not is_linux_nvidia_host()
        or approved_nvidia_boundary() is not None
    ):
        return False

    os.environ[_NVIDIA_HOST_VENV_ACK_ENV] = "1"
    os.environ[_LEGACY_NVIDIA_HOST_VENV_ACK_ENV] = "1"
    logger.warning(
        "Native Linux NVIDIA host detected on local-only Forge; approving host-venv "
        "GPU training for this process via %s=1. Remote Forge still requires an "
        "explicit secure-boundary acknowledgement.",
        _NVIDIA_HOST_VENV_ACK_ENV,
    )
    return True


def validate_security_settings(settings: ForgeSettings) -> None:
    """Fail fast on unsafe combinations; log warnings for elevated risk."""
    from forge.security.audit import audit_event
    from seiso.security.nvidia_boundary import (
        approved_nvidia_boundary,
        is_linux_nvidia_host,
        recommended_gpu_install_ack_env,
    )

    if settings.trust_proxy and not settings.trusted_proxy_ip_list:
        raise RuntimeError(
            "SEISO_TRUST_PROXY=true requires SEISO_TRUSTED_PROXY_IPS "
            "(comma-separated proxy addresses, e.g. 127.0.0.1,::1)"
        )

    if settings.allow_remote:
        if not _env_enabled(_REMOTE_ACK_ENV):
            raise RuntimeError(
                "SEISO_ALLOW_REMOTE=true requires explicit acknowledgement: "
                f"export {_REMOTE_ACK_ENV}=1"
            )
        logger.warning(
            "SEISO_ALLOW_REMOTE is enabled — Forge is exposed on the network. "
            "Use a strong password, TLS reverse proxy, and keep tools/code-exec disabled."
        )
        # Code-exec is AST deny-list + best-effort limits, not a full OS sandbox.
        # Refuse remote + code-exec entirely — no acknowledgement override.
        if settings.allow_code_exec:
            raise RuntimeError(
                "Remote access cannot be combined with code execution "
                "(AST policy is not a full OS sandbox). "
                "Disable SEISO_ALLOW_CODE_EXEC or SEISO_ALLOW_REMOTE."
            )

        tools_dangerous = settings.allow_tools or settings.allow_compat_tools
        if tools_dangerous and not _env_enabled(_REMOTE_DANGEROUS_ACK_ENV):
            raise RuntimeError(
                f"Remote access with tools requires: export {_REMOTE_DANGEROUS_ACK_ENV}=1"
            )
        if tools_dangerous:
            logger.warning("Remote access with tools enabled — high risk if credentials leak.")
        audit_event(
            "security_elevated",
            allow_remote=True,
            allow_tools=bool(settings.allow_tools),
            allow_compat_tools=bool(settings.allow_compat_tools),
            allow_code_exec=False,
            remote_dangerous_ack=_env_enabled(_REMOTE_DANGEROUS_ACK_ENV),
        )

    if settings.debug and settings.allow_remote:
        raise RuntimeError(
            "SEISO_DEBUG=true cannot be combined with SEISO_ALLOW_REMOTE=true "
            "(debug CSP allows 'unsafe-inline' and exposes /api/docs). "
            "Disable SEISO_DEBUG or SEISO_ALLOW_REMOTE."
        )
    if settings.debug:
        logger.warning("SEISO_DEBUG=true exposes /api/docs — disable in production.")

    _set_native_linux_nvidia_boundary_for_local_forge(settings)

    if is_linux_nvidia_host() and approved_nvidia_boundary() is None:
        ack = recommended_gpu_install_ack_env()
        logger.info(
            "NVIDIA GPU detected on Linux. Before GPU training, approve the secure boundary: "
            "export %s=1 (see docs/platforms/linux-nvidia.md). "
            "Hardware identifiers are not stored in job artifacts.",
            ack,
        )
