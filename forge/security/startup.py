"""Startup security policy validation and warnings."""

from __future__ import annotations

import logging
import os

from forge.config import ForgeSettings

logger = logging.getLogger(__name__)

_REMOTE_ACK_ENV = "SEISO_REMOTE_ACK"
_REMOTE_DANGEROUS_ACK_ENV = "SEISO_REMOTE_DANGEROUS_ACK"


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def validate_security_settings(settings: ForgeSettings) -> None:
    """Fail fast on unsafe combinations; log warnings for elevated risk."""
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
        dangerous = settings.allow_tools or settings.allow_code_exec or settings.allow_openai_tools
        if dangerous and not _env_enabled(_REMOTE_DANGEROUS_ACK_ENV):
            raise RuntimeError(
                "Remote access with tools or code execution requires: "
                f"export {_REMOTE_DANGEROUS_ACK_ENV}=1"
            )
        if dangerous:
            logger.warning(
                "Remote access with tools/code-exec enabled — high risk of RCE if credentials leak."
            )

    if settings.autodefense_fail_open and settings.autodefense_enabled:
        logger.warning(
            "SEISO_AUTODEFENSE_FAIL_OPEN=true — inference continues when AutoDefense is unreachable."
        )

    if settings.debug:
        logger.warning("SEISO_DEBUG=true exposes /api/docs — disable in production.")

    if is_linux_nvidia_host() and approved_nvidia_boundary() is None:
        ack = recommended_gpu_install_ack_env()
        logger.info(
            "NVIDIA GPU detected on Linux. Before GPU training, approve the secure boundary: "
            "export %s=1 (see docs/platforms/linux-nvidia.md). "
            "Hardware identifiers are not stored in job artifacts.",
            ack,
        )
