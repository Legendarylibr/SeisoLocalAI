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


def _validate_code_workspace(settings: ForgeSettings) -> None:
    from forge.services.code_workspace import (
        code_workspace_explicitly_configured,
        resolve_code_workspace,
        warn_workspace_data_overlap,
        workspace_overlaps_data_dir,
    )

    tools_enabled = (
        settings.allow_tools
        or settings.allow_code_exec
        or settings.allow_openai_tools
    )
    if tools_enabled and not code_workspace_explicitly_configured(settings):
        raise RuntimeError(
            "Tools or code execution require an explicit code workspace outside "
            "SEISO_DATA_DIR. Set SEISO_CODE_WORKSPACE to your project directory."
        )

    try:
        root = resolve_code_workspace(settings)
    except FileNotFoundError as exc:
        if tools_enabled:
            raise RuntimeError(
                "Tools or code execution require a valid SEISO_CODE_WORKSPACE directory"
            ) from exc
        logger.warning("Code workspace not configured yet: %s", exc)
        return

    if workspace_overlaps_data_dir(root, settings.data_dir):
        if tools_enabled:
            raise RuntimeError(
                f"Code workspace {root} overlaps SEISO data directory "
                f"{settings.data_dir}. Point SEISO_CODE_WORKSPACE at a separate "
                "project directory before enabling tools."
            )
        warn_workspace_data_overlap(root, settings.data_dir)
    else:
        logger.info("Seiso Code workspace: %s", root)


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
        dangerous = (
            settings.allow_tools
            or settings.allow_code_exec
            or settings.allow_openai_tools
        )
        if dangerous and not _env_enabled(_REMOTE_DANGEROUS_ACK_ENV):
            raise RuntimeError(
                "Remote access with tools or code execution requires: "
                f"export {_REMOTE_DANGEROUS_ACK_ENV}=1"
            )
        if dangerous:
            logger.warning(
                "Remote access with tools/code-exec enabled — high risk of RCE if credentials leak."
            )

    if settings.debug:
        logger.warning("SEISO_DEBUG=true exposes /api/docs — disable in production.")

    _validate_code_workspace(settings)

    _set_native_linux_nvidia_boundary_for_local_forge(settings)

    if is_linux_nvidia_host() and approved_nvidia_boundary() is None:
        ack = recommended_gpu_install_ack_env()
        logger.info(
            "NVIDIA GPU detected on Linux. Before GPU training, approve the secure boundary: "
            "export %s=1 (see docs/platforms/linux-nvidia.md). "
            "Hardware identifiers are not stored in job artifacts.",
            ack,
        )
