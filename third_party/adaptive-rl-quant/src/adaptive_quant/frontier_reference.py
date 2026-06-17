"""OpenAI-compatible frontier model client and text comparison helpers."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import sanitize_user_text
from adaptive_quant.logging_utils import read_json, write_json
from adaptive_quant.types import PromptSample

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class FrontierCompletion:
    prompt_id: str
    response_text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class LocalCompletion:
    prompt_id: str
    response_text: str
    latency_ms: float | None
    source: str


def resolve_frontier_api_key(config: FrameworkConfig) -> str:
    for env_name in (config.frontier_api_key_env, config.frontier_fallback_api_key_env):
        raw = os.environ.get(str(env_name), "").strip()
        if raw:
            return raw
    raise RuntimeError(
        "Frontier API key not found. Set "
        f"{config.frontier_api_key_env} or {config.frontier_fallback_api_key_env}."
    )


def word_overlap_score(reference_text: str, candidate_text: str) -> float:
    """Jaccard overlap on lowercase alphanumeric tokens (0..1)."""
    ref_tokens = set(_WORD_RE.findall(reference_text.lower()))
    cand_tokens = set(_WORD_RE.findall(candidate_text.lower()))
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0
    intersection = len(ref_tokens & cand_tokens)
    union = len(ref_tokens | cand_tokens)
    return float(intersection) / float(union)


def length_ratio(reference_chars: int, candidate_chars: int) -> float:
    if reference_chars <= 0:
        return 0.0 if candidate_chars > 0 else 1.0
    return float(candidate_chars) / float(reference_chars)


class FrontierReferenceClient:
    """Minimal OpenAI-compatible chat completions client (stdlib HTTP only)."""

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.api_key = resolve_frontier_api_key(config)
        self.api_base = str(config.frontier_api_base).rstrip("/")
        self.model = str(config.frontier_model).strip()
        self.timeout_s = float(config.frontier_timeout_s)

    def complete(self, prompt: PromptSample) -> FrontierCompletion:
        text = sanitize_user_text(prompt.text)
        max_chars = int(self.config.frontier_max_prompt_chars)
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": int(self.config.frontier_max_tokens),
            "temperature": float(self.config.frontier_temperature),
        }
        started = time.perf_counter()
        response = self._post_json("/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Frontier API returned no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("Frontier API choice payload is invalid")
        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Frontier API message payload is invalid")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Frontier API returned non-text content")
        usage = response.get("usage")
        prompt_tokens = None
        completion_tokens = None
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens = int(usage["prompt_tokens"])
            if isinstance(usage.get("completion_tokens"), int):
                completion_tokens = int(usage["completion_tokens"])
        return FrontierCompletion(
            prompt_id=prompt.prompt_id,
            response_text=content.strip(),
            model=str(response.get("model") or self.model),
            latency_ms=float(latency_ms),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_base}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Frontier API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Frontier API request failed: {exc}") from exc
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Frontier API returned non-object JSON")
        return parsed


def load_frontier_reference(path: str) -> dict[str, dict[str, Any]]:
    payload = read_json(path, label="Frontier reference file")
    records = normalize_frontier_reference_payload(payload)
    if not records:
        raise ValueError(f"Frontier reference file has no prompt records: {path}")
    return records


def save_frontier_reference(
    path: str, records: dict[str, dict[str, Any]], *, meta: dict[str, Any]
) -> None:
    payload = {"meta": meta, "prompts": records}
    write_json(path, payload)


def frontier_record_from_completion(completion: FrontierCompletion) -> dict[str, Any]:
    return {
        "response_text": completion.response_text,
        "model": completion.model,
        "latency_ms": completion.latency_ms,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
    }


def normalize_frontier_reference_payload(payload: object) -> dict[str, dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("prompts"), dict):
        prompts = payload["prompts"]
        records: dict[str, dict[str, Any]] = {}
        for prompt_id, value in prompts.items():
            if isinstance(prompt_id, str) and isinstance(value, dict):
                records[prompt_id] = dict(value)
        return records
    if isinstance(payload, dict):
        records = {}
        for prompt_id, value in payload.items():
            if prompt_id == "meta":
                continue
            if isinstance(prompt_id, str) and isinstance(value, dict):
                records[prompt_id] = dict(value)
        return records
    raise TypeError("Unsupported frontier reference payload")


__all__ = [
    "FrontierCompletion",
    "FrontierReferenceClient",
    "LocalCompletion",
    "frontier_record_from_completion",
    "length_ratio",
    "load_frontier_reference",
    "normalize_frontier_reference_payload",
    "resolve_frontier_api_key",
    "save_frontier_reference",
    "word_overlap_score",
]
