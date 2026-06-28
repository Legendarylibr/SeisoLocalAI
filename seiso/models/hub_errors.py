"""Shared Hugging Face Hub error messages for search, download, and connectivity."""

from __future__ import annotations

from typing import Literal

HubErrorContext = Literal["search", "download", "probe"]


def is_gated_hub_error(exc: BaseException) -> bool:
    """True when Hub denied access (gated repo, missing/invalid token for private model)."""
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code in (401, 403):
        return True
    msg = str(exc).strip().lower()
    return any(
        token in msg
        for token in (
            "gated",
            "authorized list",
            "not in the authorized",
            "access denied",
            "403",
            "401",
        )
    )


def is_hub_transport_error(exc: BaseException) -> bool:
    """True for network/proxy/DNS failures — not Hub HTTP 4xx/5xx (HfHubHTTPError)."""
    if isinstance(exc, OSError):
        return True
    module = exc.__class__.__module__
    if module.startswith(("httpx", "httpcore", "urllib3")):
        return True
    return module == "requests.exceptions"


def format_hub_error(
    exc: Exception,
    *,
    context: HubErrorContext = "probe",
    repo_id: str | None = None,
    status_code: int | None = None,
) -> str:
    """Return a user-facing Hub error string."""
    msg = str(exc).strip() or exc.__class__.__name__
    lowered = msg.lower()
    code = status_code or getattr(getattr(exc, "response", None), "status_code", None)

    if (
        code == 401
        or code == 403
        or "401" in msg
        or "403" in msg
        or "gated" in lowered
        or "authorized" in lowered
    ):
        if context == "download" and repo_id:
            if "authorized" in lowered or "gated" in lowered:
                return (
                    f"Access denied for {repo_id}. Open https://huggingface.co/{repo_id}, "
                    "sign in, and accept the model license — then retry. "
                    "You can also save a Hugging Face token in Settings or run `hf auth login`."
                )
            return (
                f"Access denied for {repo_id}. This model may be gated — "
                "save a Hugging Face token in Settings or run `hf auth login`."
            )
        return (
            "Hugging Face Hub access denied. Save a token in Settings or run `hf auth login` "
            "to access gated models."
        )

    if (
        code == 429
        or "429" in msg
        or "rate limit" in lowered
        or "too many requests" in lowered
    ):
        if context == "download" and repo_id:
            return (
                f"Hugging Face anonymous API rate limit reached while downloading {repo_id}. "
                "Public models do not require a token, but anonymous requests are throttled. "
                "Wait a few minutes and retry, or add a free HF token for higher limits."
            )
        return (
            "Hugging Face Hub rate limit reached. "
            "Wait a few minutes and retry, or add a free HF token for higher limits."
        )

    if code == 404 or "404" in msg or "not found" in lowered:
        if context == "download" and repo_id:
            return f"Model repo not found on Hugging Face Hub: {repo_id}"
        if context == "search":
            return "Hugging Face Hub search endpoint not found."
        return msg

    if "proxy" in lowered:
        return "Network proxy blocked Hugging Face Hub. Check proxy settings and try again."

    if "connection" in lowered or "network" in lowered or "resolve" in lowered:
        if context == "download" and repo_id:
            return f"Cannot reach huggingface.co while downloading {repo_id}. Check your network."
        if context == "search":
            return "Cannot reach huggingface.co. Check your network and try again."
        return f"Cannot reach huggingface.co — check your network connection. ({msg})"

    if "timeout" in lowered or "timed out" in lowered:
        if context == "download" and repo_id:
            return f"Download timed out for {repo_id}. Retry or set HF_HUB_DOWNLOAD_TIMEOUT higher."
        if context == "search":
            return "Hugging Face Hub search timed out. Try again in a moment."
        return f"Hugging Face Hub timed out — try again or increase HF_HUB_DOWNLOAD_TIMEOUT. ({msg})"

    if context == "download" and repo_id:
        return f"Hub download failed for {repo_id}: {msg}"
    if context == "search":
        return f"Hugging Face Hub search failed: {msg}"
    return msg
