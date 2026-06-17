"""Hugging Face Hub preflight checks — validate metadata and repo availability before export."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from seiso.export.model_card import HubModelMetadata, render_readme

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_HF_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALLOWED_LICENSES = frozenset(
    {
        "apache-2.0",
        "mit",
        "llama2",
        "llama3",
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "gemma",
        "openrail",
        "bigscience-openrail-m",
        "bigcode-openrail-m",
        "afl-3.0",
        "artistic-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc-by-nc-4.0",
        "cc-by-nc-sa-4.0",
        "cc0-1.0",
        "gpl-3.0",
        "lgpl-3.0",
        "other",
    }
)


@dataclass
class HubPrecheckResult:
    """Structured report from a Hub export preflight."""

    repo_id: str
    ok: bool
    token_valid: bool = False
    token_username: str | None = None
    repo_exists: bool = False
    repo_available: bool = False
    repo_owned_by_user: bool = False
    metadata_valid: bool = False
    model_card_preview: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "ok": self.ok,
            "token_valid": self.token_valid,
            "token_username": self.token_username,
            "repo_exists": self.repo_exists,
            "repo_available": self.repo_available,
            "repo_owned_by_user": self.repo_owned_by_user,
            "metadata_valid": self.metadata_valid,
            "model_card_preview": self.model_card_preview,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_repo_id(repo_id: str) -> None:
    """Validate Hugging Face repo id format (org/name)."""
    normalized = repo_id.strip()
    if not normalized:
        raise ValueError("Repo id is required")
    if "\x00" in normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError("Repo id contains invalid control characters")
    if ".." in normalized.split("/"):
        raise ValueError("Repo id must not contain '..'")
    if not _HF_REPO_ID_RE.match(normalized):
        raise ValueError(
            f"Invalid repo id {normalized!r}: expected '<org>/<name>' with alphanumeric, '.', '_', or '-' only"
        )
    org, name = normalized.split("/", 1)
    if not _HF_SEGMENT_RE.match(org) or not _HF_SEGMENT_RE.match(name):
        raise ValueError(f"Invalid repo id segments in {normalized!r}")


def validate_hub_metadata(meta: HubModelMetadata, *, formats: list[str] | None = None) -> list[str]:
    """Return validation errors for Hub metadata and model card."""
    errors: list[str] = []
    try:
        meta.validate()
    except ValueError as exc:
        errors.append(str(exc))

    try:
        validate_repo_id(meta.repo_id)
    except ValueError as exc:
        errors.append(str(exc))

    if meta.license.strip().lower() not in _ALLOWED_LICENSES:
        errors.append(
            f"License {meta.license!r} is not in the supported list; use a standard SPDX id (e.g. apache-2.0)"
        )

    if len(meta.model_name) > 96:
        errors.append("Model name must be 96 characters or fewer")

    if formats:
        fmt_set = {f.lower() for f in formats}
        if "gguf" in fmt_set and not meta.quantizations:
            errors.append("GGUF export requires at least one quantization in metadata")

    try:
        card = render_readme(meta)
        if not card.startswith("---"):
            errors.append("Model card must include YAML frontmatter")
        if len(card) > 500_000:
            errors.append("Model card exceeds Hugging Face size limits")
    except Exception as exc:
        errors.append(f"Model card generation failed: {exc}")

    return errors


def precheck_hub_export(
    *,
    repo_id: str,
    token: str,
    metadata: HubModelMetadata | None = None,
    formats: list[str] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> HubPrecheckResult:
    """
    Validate token, repo availability, and model card before starting a heavy export.

    Fails fast when the repo is taken by another user or metadata is invalid.
    """
    result = HubPrecheckResult(repo_id=repo_id.strip(), ok=False)

    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    try:
        validate_repo_id(result.repo_id)
    except ValueError as exc:
        result.errors.append(str(exc))
        return result

    if metadata:
        meta_errors = validate_hub_metadata(metadata, formats=formats)
        result.errors.extend(meta_errors)
        if not meta_errors:
            result.metadata_valid = True
            result.model_card_preview = render_readme(metadata)
    else:
        result.warnings.append("No metadata provided — model card will not be written")

    if not token or not token.strip():
        result.errors.append("Hugging Face token is required for Hub upload")
        return result

    api = HfApi(token=token.strip())

    try:
        who = api.whoami()
        result.token_valid = True
        result.token_username = who.get("name") or who.get("fullname")
        log(f"Token valid for user: {result.token_username}")
    except Exception as exc:
        result.errors.append(f"Invalid Hugging Face token: {exc}")
        return result

    if metadata and result.token_username:
        expected_org = metadata.username.strip().lower()
        token_org = result.token_username.strip().lower()
        if expected_org != token_org:
            result.warnings.append(
                f"Metadata username {metadata.username!r} differs from token account {result.token_username!r}; "
                "repo will be created under your token account namespace"
            )

    try:
        info = api.repo_info(result.repo_id, repo_type="model")
        result.repo_exists = True
        author = getattr(info, "author", None) or getattr(info, "id", "").split("/")[0]
        if result.token_username and author and author.lower() == result.token_username.lower():
            result.repo_owned_by_user = True
            result.repo_available = True
            result.warnings.append(f"Repo {result.repo_id} already exists — upload will add a new revision")
        else:
            result.repo_available = False
            result.errors.append(
                f"Repo {result.repo_id} already exists and is owned by {author!r}, not your account"
            )
    except HfHubHTTPError as exc:
        if exc.response.status_code == 404:
            result.repo_available = True
            log(f"Repo {result.repo_id} is available")
        else:
            result.errors.append(f"Hub repo lookup failed: {exc}")
    except Exception as exc:
        result.errors.append(f"Hub repo lookup failed: {exc}")

    result.ok = result.token_valid and result.repo_available and not result.errors
    if result.ok:
        log(f"Hub precheck passed for {result.repo_id}")
    return result


def assert_hub_precheck_ok(result: HubPrecheckResult) -> None:
    """Raise ValueError with all precheck errors if export should not proceed."""
    if result.ok:
        return
    detail = "; ".join(result.errors) if result.errors else "Hub precheck failed"
    raise ValueError(detail)
