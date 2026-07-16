"""HTTP helpers and engine URL resolution for slime rollout backends."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from seiso.slime.config import SingleGpuSlimeConfig


def _http_json_request(
    *,
    base_url: str,
    path: str,
    method: str,
    body: dict[str, Any] | None,
    api_key: str,
    timeout_s: float,
    label: str,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{label} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label} request failed: {exc}") from exc
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # vLLM LoRA endpoints may return plain text "Success: ..."
        return {"result": raw.strip()}
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


def sglang_engine_urls(config: SingleGpuSlimeConfig) -> list[str]:
    """Resolve one or more SGLang engine base URLs (comma-separated or multi field)."""
    urls: list[str] = []
    primary = str(getattr(config, "sglang_base_url", "") or "").strip()
    if primary:
        urls.extend(part.strip() for part in primary.split(",") if part.strip())
    extra = getattr(config, "sglang_engine_urls", None) or []
    if isinstance(extra, str):
        urls.extend(part.strip() for part in extra.split(",") if part.strip())
    elif isinstance(extra, (list, tuple)):
        urls.extend(str(u).strip() for u in extra if str(u).strip())
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        key = url.rstrip("/")
        _validate_sglang_url(key)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _validate_sglang_url(url: str) -> None:
    """Reject non-HTTP(S) schemes (basic SSRF hardening for config-controlled URLs)."""
    _validate_http_engine_url(url, label="sglang")


def _validate_http_engine_url(url: str, *, label: str = "engine") -> None:
    """Reject non-HTTP(S) schemes (basic SSRF hardening for config-controlled URLs)."""
    lowered = url.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError(f"{label} URL must use http:// or https:// scheme, got {url!r}")


def _strip_openai_v1_suffix(url: str) -> str:
    """Normalize ``.../v1`` base URLs used by managed vLLM to engine host roots."""
    cleaned = url.rstrip("/")
    if cleaned.lower().endswith("/v1"):
        return cleaned[:-3]
    return cleaned


def resolve_vllm_base_url(config: SingleGpuSlimeConfig) -> str:
    """Resolve vLLM engine URL from config, managed multi-GPU state, or env."""
    engines = vllm_engine_urls(config, allow_empty_primary=True)
    if engines:
        return engines[0]
    # Adopt Seiso-managed multi-GPU vLLM when it is already running.
    try:
        from seiso.inference.managed_vllm import get_status

        status = get_status()
        if status.get("running") and status.get("base_url"):
            return _strip_openai_v1_suffix(str(status["base_url"]))
    except Exception:
        pass
    try:
        from seiso.env import env_int, env_str

        host = (env_str("SEISO_MANAGED_VLLM_HOST", "127.0.0.1") or "127.0.0.1").strip()
        port = int(env_int("SEISO_MANAGED_VLLM_PORT", 0) or 0)
        if port > 0:
            return f"http://{host}:{port}"
    except Exception:
        pass
    return ""


def vllm_engine_urls(
    config: SingleGpuSlimeConfig,
    *,
    allow_empty_primary: bool = False,
) -> list[str]:
    """Resolve one or more vLLM engine base URLs (comma-separated or multi field)."""
    urls: list[str] = []
    primary = str(getattr(config, "vllm_base_url", "") or "").strip()
    if primary:
        urls.extend(part.strip() for part in primary.split(",") if part.strip())
    extra = getattr(config, "vllm_engine_urls", None) or []
    if isinstance(extra, str):
        urls.extend(part.strip() for part in extra.split(",") if part.strip())
    elif isinstance(extra, (list, tuple)):
        urls.extend(str(u).strip() for u in extra if str(u).strip())
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        key = _strip_openai_v1_suffix(url)
        _validate_http_engine_url(key, label="vllm")
        if key not in seen:
            seen.add(key)
            out.append(key)
    if not out and not allow_empty_primary:
        raise ValueError("vllm_base_url is required for vLLM rollout")
    return out
