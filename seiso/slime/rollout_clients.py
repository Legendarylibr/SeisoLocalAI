"""SGLang / vLLM OpenAI-compatible rollout clients."""

from __future__ import annotations

import contextlib
from typing import Any, NamedTuple

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.rollout_http import (
    _http_json_request,
    _strip_openai_v1_suffix,
    _validate_http_engine_url,
    _validate_sglang_url,
    resolve_vllm_base_url,
    sglang_engine_urls,
    vllm_engine_urls,
)


class HttpCompletion(NamedTuple):
    """One remote completion: text, optional engine ids, OpenAI finish_reason."""

    text: str
    token_ids: list[int] | None
    finish_reason: str | None


def _extract_completion_token_ids(
    choice: dict[str, Any],
    payload: dict[str, Any],
) -> list[int] | None:
    """Best-effort parse of token ids from OpenAI/SGLang completion payloads."""
    for key in ("token_ids", "output_ids", "output_token_ids", "tokens"):
        value = choice.get(key)
        if isinstance(value, list) and value and all(isinstance(x, int) for x in value):
            return [int(x) for x in value]
    meta = choice.get("meta_info") or payload.get("meta_info") or {}
    if isinstance(meta, dict):
        for key in ("output_token_ids", "output_ids", "token_ids"):
            value = meta.get(key)
            if isinstance(value, list) and value and all(isinstance(x, int) for x in value):
                return [int(x) for x in value]
    # logprobs.tokens may be strings ("token") or ids depending on engine.
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        tokens = logprobs.get("token_ids") or logprobs.get("tokens")
        if isinstance(tokens, list) and tokens and all(isinstance(x, int) for x in tokens):
            return [int(x) for x in tokens]
    return None


def _extract_finish_reason(choice: dict[str, Any]) -> str | None:
    raw = choice.get("finish_reason")
    if raw is None:
        raw = choice.get("stop_reason")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def _normalize_rollout_finish_status(finish_reason: str | None) -> str | None:
    """Map OpenAI/vLLM/SGLang finish_reason → slime status, or None if unknown."""
    if not finish_reason:
        return None
    key = str(finish_reason).strip().lower()
    if key in {"stop", "eos", "end_turn", "stop_sequence", "tool_calls"}:
        return "stop"
    if key in {"length", "max_tokens", "model_length", "max_length"}:
        return "length"
    return None


def _completions_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_token_ids: bool = False,
) -> dict[str, Any]:
    """OpenAI completions body; optional engine-specific token-id knobs."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
    }
    if request_token_ids:
        # Best-effort: some vLLM/SGLang builds return sampled ids when asked.
        payload["return_tokens_as_token_ids"] = True
    return payload


def _post_completion(
    post_json,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    label: str,
) -> HttpCompletion:
    """POST /v1/completions; retry without token-id knobs if the engine rejects them."""
    attempts = (True, False)
    last_exc: Exception | None = None
    for request_ids in attempts:
        payload = _completions_payload(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            request_token_ids=request_ids,
        )
        try:
            data = post_json("/v1/completions", payload)
        except RuntimeError as exc:
            last_exc = exc
            detail = str(exc).lower()
            if request_ids and (
                "400" in detail
                or "unknown" in detail
                or "unexpected" in detail
                or "invalid" in detail
            ):
                continue
            raise
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"{label} /v1/completions returned no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError(f"{label} choice payload is invalid")
        text = first.get("text")
        if not isinstance(text, str):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                text = message["content"]
            else:
                raise RuntimeError(f"{label} choice missing text")
        return HttpCompletion(
            text=text,
            token_ids=_extract_completion_token_ids(first, data),
            finish_reason=_extract_finish_reason(first),
        )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} /v1/completions failed")


class SGLangRolloutClient:
    """Minimal OpenAI-compatible client for SGLang ``/v1/completions``."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 120.0,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_tokens: int = 256,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    @classmethod
    def from_config(cls, config: SingleGpuSlimeConfig) -> SGLangRolloutClient:
        engines = sglang_engine_urls(config)
        if not engines:
            raise ValueError("sglang_base_url is required for SGLang rollout")
        # Generation uses the first engine; weight sync fans out to all URLs.
        base = engines[0]
        _validate_sglang_url(base)
        model = str(getattr(config, "sglang_model", "") or "").strip() or config.model_id
        return cls(
            base_url=base,
            model=model,
            api_key=str(getattr(config, "sglang_api_key", "EMPTY") or "EMPTY"),
            timeout_s=float(getattr(config, "sglang_timeout_s", 120.0) or 120.0),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            max_tokens=int(config.max_new_tokens),
        )

    def complete(self, prompt: str) -> str:
        return self.complete_http(prompt).text

    def complete_with_tokens(self, prompt: str) -> tuple[str, list[int] | None]:
        """Back-compat: ``(text, token_ids)``. Prefer ``complete_http``."""
        result = self.complete_http(prompt)
        return result.text, result.token_ids

    def complete_http(self, prompt: str) -> HttpCompletion:
        """Return text, optional engine token ids, and finish_reason."""
        return _post_completion(
            self._post_json,
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            label="SGLang",
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        *,
        weight_version: str | None = None,
    ) -> dict[str, Any]:
        """Hot-reload HF weights (slime / SGLang ``update_weights_from_disk``)."""
        payload: dict[str, Any] = {"model_path": model_path}
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._post_json("/update_weights_from_disk", payload)

    def pull_weights(
        self,
        *,
        local_checkpoint_dir: str,
        source_dir: str,
        target_version: int,
    ) -> dict[str, Any]:
        """Slime-patched engines: apply disk delta then local reload.

        Vanilla SGLang does not implement this endpoint — callers must fall back
        to ``update_weights_from_disk`` with a full HF checkpoint.
        """
        return self._post_json(
            "/pull_weights",
            {
                "local_checkpoint_dir": local_checkpoint_dir,
                "source_dir": source_dir,
                "target_version": int(target_version),
            },
        )

    def flush_cache(self) -> dict[str, Any] | None:
        """Best-effort KV cache flush after a weight update."""
        try:
            return self._request("GET", "/flush_cache", body=None)
        except RuntimeError:
            return None

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body=payload)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return _http_json_request(
            base_url=self.base_url,
            path=path,
            method=method,
            body=body,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            label="SGLang",
        )


class VLLMRolloutClient:
    """OpenAI-compatible client for vLLM multi-GPU rollouts + weight hot-reload."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 120.0,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_tokens: int = 256,
        lora_name: str = "seiso_slime_policy",
    ) -> None:
        self.base_url = _strip_openai_v1_suffix(base_url.rstrip("/"))
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.lora_name = lora_name
        self._active_model = model

    @classmethod
    def from_config(cls, config: SingleGpuSlimeConfig) -> VLLMRolloutClient:
        engines = vllm_engine_urls(config, allow_empty_primary=True)
        base = engines[0] if engines else resolve_vllm_base_url(config)
        if not base:
            raise ValueError("vllm_base_url is required for vLLM rollout")
        _validate_http_engine_url(base, label="vllm")
        model = str(getattr(config, "vllm_model", "") or "").strip() or config.model_id
        lora_name = (
            str(getattr(config, "vllm_lora_name", "") or "seiso_slime_policy").strip()
            or "seiso_slime_policy"
        )
        client = cls(
            base_url=base,
            model=model,
            api_key=str(getattr(config, "vllm_api_key", "EMPTY") or "EMPTY"),
            timeout_s=float(getattr(config, "vllm_timeout_s", 120.0) or 120.0),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            max_tokens=int(config.max_new_tokens),
            lora_name=lora_name,
        )
        # After LoRA weight sync, the engine serves under the dynamic adapter name.
        mode = str(getattr(config, "vllm_weight_mode", "auto") or "auto").lower()
        if bool(getattr(config, "vllm_sync_weights", True)) and (
            mode == "lora" or (mode == "auto" and bool(getattr(config, "use_lora", False)))
        ):
            client.use_lora_model(True)
        return client

    def complete(self, prompt: str) -> str:
        return self.complete_http(prompt).text

    def complete_with_tokens(self, prompt: str) -> tuple[str, list[int] | None]:
        """Back-compat: ``(text, token_ids)``. Prefer ``complete_http``."""
        result = self.complete_http(prompt)
        return result.text, result.token_ids

    def complete_http(self, prompt: str) -> HttpCompletion:
        """Return text, optional engine token ids, and finish_reason."""
        return _post_completion(
            self._post_json,
            model=self._active_model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            label="vLLM",
        )

    def use_lora_model(self, enabled: bool = True) -> None:
        """Route completions to the dynamic LoRA name after a successful load."""
        self._active_model = self.lora_name if enabled else self.model

    def load_lora_adapter(self, lora_path: str, *, lora_name: str | None = None) -> dict[str, Any]:
        """Hot-load a PEFT adapter via vLLM ``/v1/load_lora_adapter``."""
        name = (lora_name or self.lora_name).strip() or self.lora_name
        # Unload first so reloads replace the previous step's adapter.
        with contextlib.suppress(RuntimeError):
            self.unload_lora_adapter(lora_name=name)
        result = self._post_json(
            "/v1/load_lora_adapter",
            {"lora_name": name, "lora_path": lora_path},
        )
        self.lora_name = name
        self.use_lora_model(True)
        return result

    def unload_lora_adapter(self, *, lora_name: str | None = None) -> dict[str, Any]:
        name = (lora_name or self.lora_name).strip() or self.lora_name
        result = self._post_json(
            "/v1/unload_lora_adapter",
            {"lora_name": name},
        )
        if self._active_model == name:
            self.use_lora_model(False)
        return result

    def update_weights_from_disk(
        self,
        model_path: str,
        *,
        weight_version: str | None = None,
    ) -> dict[str, Any]:
        """Best-effort full weight reload (SGLang-compatible or custom endpoint)."""
        payload: dict[str, Any] = {"model_path": model_path}
        if weight_version is not None:
            payload["weight_version"] = weight_version
        # Prefer documented-ish paths; first success wins.
        errors: list[str] = []
        for path in (
            "/update_weights_from_disk",
            "/v1/update_weights_from_disk",
            "/reload_weights",
        ):
            try:
                return self._post_json(path, payload)
            except RuntimeError as exc:
                errors.append(f"{path}: {exc}")
        raise RuntimeError("vLLM full weight reload failed on all endpoints: " + "; ".join(errors))

    def pause(self) -> dict[str, Any] | None:
        with contextlib.suppress(RuntimeError):
            return self._post_json("/pause", {})
        return None

    def resume(self) -> dict[str, Any] | None:
        with contextlib.suppress(RuntimeError):
            return self._post_json("/resume", {})
        return None

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body=payload)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return _http_json_request(
            base_url=self.base_url,
            path=path,
            method=method,
            body=body,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            label="vLLM",
        )
