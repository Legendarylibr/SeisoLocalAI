"""Online rollout backends for slime-style GRPO.

* ``hf`` / ``data_gen`` — colocated Hugging Face ``generate`` (default; single-GPU)
* ``sglang`` — OpenAI-compatible HTTP generation against a running SGLang server
* ``vllm`` — OpenAI-compatible HTTP generation against a running vLLM server
  (multi-GPU tensor-parallel rollouts; pairs with managed multi-GPU vLLM)

Completions are always produced online. Prompt corpora (labels/tests) come from
JSONL or high-level ``data_gen`` corpus materialization — never from stored
model outputs.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from seiso.slime_single_gpu.config import SingleGpuSlimeConfig

# Canonical: hf | sglang | vllm | auto. "data_gen" is a legacy alias of "hf".
_ROLLOUT_BACKENDS = frozenset({"hf", "sglang", "vllm", "auto", "data_gen"})
_HTTP_ROLLOUT_BACKENDS = frozenset({"sglang", "vllm"})


def _normalize_backend_name(name: str) -> str:
    key = str(name or "hf").lower().strip()
    if key == "data_gen":
        return "hf"
    return key


def resolve_rollout_backend(
    config: SingleGpuSlimeConfig,
    *,
    world_size: int = 1,
) -> str:
    """Resolve effective backend.

    * ``hf`` (default; alias ``data_gen``) — colocated Hugging Face generate
    * ``sglang`` — OpenAI-compatible SGLang HTTP
    * ``vllm`` — OpenAI-compatible vLLM HTTP (multi-GPU TP server)
    * ``auto`` — prefer vLLM then SGLang when a base URL is set and
      ``world_size > 1``; otherwise HF
    """
    name = _normalize_backend_name(getattr(config, "rollout_backend", "hf") or "hf")
    if name not in {"hf", "sglang", "vllm", "auto"}:
        raise ValueError(
            f"rollout_backend must be one of: hf, sglang, vllm, auto (got {name!r})"
        )
    if name == "auto":
        if world_size > 1:
            if resolve_vllm_base_url(config):
                return "vllm"
            if str(getattr(config, "sglang_base_url", "") or "").strip():
                return "sglang"
        return "hf"
    return name


def validate_rollout_backend_config(config: SingleGpuSlimeConfig) -> None:
    name = _normalize_backend_name(getattr(config, "rollout_backend", "hf") or "hf")
    if name not in {"hf", "sglang", "vllm", "auto"}:
        raise ValueError(
            f"rollout_backend must be one of: hf, sglang, vllm, auto (got {name!r})"
        )
    if name == "sglang":
        base = str(getattr(config, "sglang_base_url", "") or "").strip()
        if not base:
            raise ValueError(
                "rollout_backend=sglang requires sglang_base_url "
                "(e.g. http://127.0.0.1:30000)"
            )
    if name == "vllm":
        base = resolve_vllm_base_url(config)
        if not base:
            raise ValueError(
                "rollout_backend=vllm requires vllm_base_url "
                "(e.g. http://127.0.0.1:8000), or a running managed multi-GPU "
                "vLLM server (SEISO_MANAGED_VLLM_ENABLED=true)"
            )
    timeout = float(getattr(config, "sglang_timeout_s", 120.0) or 120.0)
    if timeout <= 0:
        raise ValueError("sglang_timeout_s must be positive")
    max_workers = int(getattr(config, "sglang_max_workers", 8) or 8)
    if max_workers < 1:
        raise ValueError("sglang_max_workers must be positive")
    vllm_timeout = float(getattr(config, "vllm_timeout_s", 120.0) or 120.0)
    if vllm_timeout <= 0:
        raise ValueError("vllm_timeout_s must be positive")
    vllm_workers = int(getattr(config, "vllm_max_workers", 8) or 8)
    if vllm_workers < 1:
        raise ValueError("vllm_max_workers must be positive")
    mode = str(getattr(config, "vllm_weight_mode", "auto") or "auto").lower()
    if mode not in {"auto", "lora", "full"}:
        raise ValueError("vllm_weight_mode must be one of: auto, lora, full")
    if int(getattr(config, "vllm_weight_keep", 2) or 2) < 1:
        raise ValueError("vllm_weight_keep must be >= 1")

@dataclass(frozen=True)
class GeneratedChunk:
    """One generation chunk: parallel lists aligned as prompt_idx * n + k."""

    prompts: list[str]
    completions: list[str]
    # When set (HF generate), full sequences; else None → re-tokenize or use token ids.
    sequences: Any | None = None
    prompt_width: int | None = None
    # Optional per-completion token ids from SGLang when the server provides them.
    completion_token_ids: list[list[int] | None] | None = None


def _as_chat_messages(prompt: str | list[Any]) -> list[dict[str, str]]:
    """Normalize slime chat prompts or plain strings to OpenAI-style messages."""
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for item in prompt:
            if isinstance(item, dict) and "content" in item:
                role = str(item.get("role") or "user")
                messages.append({"role": role, "content": str(item["content"])})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages or [{"role": "user", "content": ""}]
    return [{"role": "user", "content": str(prompt)}]


def format_generation_prompt(
    tokenizer,
    prompt: str | list[Any],
    config: SingleGpuSlimeConfig,
) -> str:
    """Apply optional chat template (slime --apply-chat-template), then thinking open."""
    messages = _as_chat_messages(prompt)
    # Optional thinking instruction on the last user turn only.
    if config.require_thinking_trace:
        last = messages[-1]
        content = last["content"]
        if "<think>" not in content.lower():
            last = {
                **last,
                "content": (f"{content.rstrip()}\n\n{config.thinking_instruction}\n<think>"),
            }
            messages = [*messages[:-1], last]

    use_chat = bool(getattr(config, "apply_chat_template", True))
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    # Fallback: concatenate message contents (no template).
    return "\n".join(m["content"] for m in messages)


def generate_data_gen_chunk(
    *,
    generation_model,
    tokenizer,
    prompts: list[str],
    config: SingleGpuSlimeConfig,
    torch,
) -> GeneratedChunk:
    """Colocated HF generate — preserves prior single-GPU rollout behavior."""
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_tokens,
    ).to(config.device)
    prompt_width = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        generated = generation_model.generate(
            **encoded,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_return_sequences=config.rollouts_per_prompt,
        )
    completions = tokenizer.batch_decode(
        generated[:, prompt_width:],
        skip_special_tokens=True,
    )
    # Expand prompts to match num_return_sequences layout.
    expanded_prompts: list[str] = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * config.rollouts_per_prompt)
    return GeneratedChunk(
        prompts=expanded_prompts,
        completions=list(completions),
        sequences=generated,
        prompt_width=prompt_width,
    )


def generate_sglang_chunk(
    *,
    tokenizer,
    prompts: list[str],
    config: SingleGpuSlimeConfig,
) -> GeneratedChunk:
    """Generate ``rollouts_per_prompt`` completions per prompt via SGLang HTTP."""
    del tokenizer  # prompts are already formatted strings
    client = SGLangRolloutClient.from_config(config)
    return _generate_http_chunk(
        client=client,
        prompts=prompts,
        rollouts_per_prompt=config.rollouts_per_prompt,
        max_workers=int(getattr(config, "sglang_max_workers", 8) or 8),
    )


def generate_vllm_chunk(
    *,
    tokenizer,
    prompts: list[str],
    config: SingleGpuSlimeConfig,
) -> GeneratedChunk:
    """Generate ``rollouts_per_prompt`` completions per prompt via vLLM HTTP."""
    del tokenizer  # prompts are already formatted strings
    client = VLLMRolloutClient.from_config(config)
    return _generate_http_chunk(
        client=client,
        prompts=prompts,
        rollouts_per_prompt=config.rollouts_per_prompt,
        max_workers=int(getattr(config, "vllm_max_workers", 8) or 8),
    )


def _generate_http_chunk(
    *,
    client: Any,
    prompts: list[str],
    rollouts_per_prompt: int,
    max_workers: int,
) -> GeneratedChunk:
    """Shared OpenAI ``/v1/completions`` fan-out for SGLang and vLLM."""
    n = int(rollouts_per_prompt)
    jobs: list[tuple[int, str]] = []
    for p_idx, prompt in enumerate(prompts):
        for _ in range(n):
            jobs.append((p_idx, prompt))

    results: list[tuple[str, list[int] | None] | None] = [None] * len(jobs)
    workers = min(max(1, int(max_workers)), max(1, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(client.complete_with_tokens, prompt): idx
            for idx, (_p_idx, prompt) in enumerate(jobs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    completions: list[str] = []
    token_id_lists: list[list[int] | None] = []
    for item in results:
        if item is None:
            completions.append("")
            token_id_lists.append(None)
        else:
            text, tids = item
            completions.append(text)
            token_id_lists.append(tids)
    expanded_prompts = [prompt for _p_idx, prompt in jobs]
    return GeneratedChunk(
        prompts=expanded_prompts,
        completions=completions,
        sequences=None,
        prompt_width=None,
        completion_token_ids=token_id_lists,
    )

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
    return None


def build_sequence_tensors(
    *,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    config: SingleGpuSlimeConfig,
    torch,
    device: str,
    completion_token_ids: list[list[int] | None] | None = None,
) -> list[dict[str, Any]]:
    """Tokenize prompt+completion pairs into per-row rollout tensors.

    Prefer server-provided ``completion_token_ids`` when present (avoids BPE
    mismatch). Otherwise retokenize text with the same prompt string the
    server saw (``add_special_tokens=False``).
    """
    rows: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id
    for idx, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=config.max_prompt_tokens,
            return_tensors="pt",
        )["input_ids"][0]
        server_ids = None
        if completion_token_ids is not None and idx < len(completion_token_ids):
            server_ids = completion_token_ids[idx]
        if server_ids is not None and len(server_ids) > 0:
            comp_ids = torch.tensor(server_ids[: config.max_new_tokens], dtype=torch.long)
        elif completion:
            comp_ids = tokenizer(
                completion,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_new_tokens,
                return_tensors="pt",
            )["input_ids"][0]
        else:
            comp_ids = torch.zeros(0, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, comp_ids], dim=0).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        prompt_len = int(prompt_ids.numel())
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        if prompt_len < int(input_ids.numel()):
            resp = input_ids[prompt_len:]
            resp_mask = torch.ones_like(resp, dtype=torch.bool)
            if pad_id is not None and eos_id is not None and pad_id == eos_id:
                eos_hits = (resp == eos_id).nonzero(as_tuple=False)
                if eos_hits.numel() > 0:
                    first = int(eos_hits[0].item())
                    resp_mask[first + 1 :] = False
            elif pad_id is not None:
                resp_mask = resp != pad_id
            response_mask[prompt_len:] = resp_mask
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "prompt_len": prompt_len,
            }
        )
    return rows


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
        text, _token_ids = self.complete_with_tokens(prompt)
        return text

    def complete_with_tokens(self, prompt: str) -> tuple[str, list[int] | None]:
        """Return (text, optional engine token ids when the server provides them)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
        }
        data = self._post_json("/v1/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("SGLang /v1/completions returned no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("SGLang choice payload is invalid")
        text = first.get("text")
        if not isinstance(text, str):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                text = message["content"]
            else:
                raise RuntimeError("SGLang choice missing text")
        token_ids = _extract_completion_token_ids(first, data)
        return text, token_ids

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
        lora_name = str(
            getattr(config, "vllm_lora_name", "") or "seiso_slime_policy"
        ).strip() or "seiso_slime_policy"
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
            mode == "lora"
            or (mode == "auto" and bool(getattr(config, "use_lora", False)))
        ):
            client.use_lora_model(True)
        return client
    def complete(self, prompt: str) -> str:
        text, _token_ids = self.complete_with_tokens(prompt)
        return text

    def complete_with_tokens(self, prompt: str) -> tuple[str, list[int] | None]:
        """Return (text, optional engine token ids when the server provides them)."""
        payload = {
            "model": self._active_model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
        }
        data = self._post_json("/v1/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("vLLM /v1/completions returned no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("vLLM choice payload is invalid")
        text = first.get("text")
        if not isinstance(text, str):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                text = message["content"]
            else:
                raise RuntimeError("vLLM choice missing text")
        token_ids = _extract_completion_token_ids(first, data)
        return text, token_ids

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
        raise RuntimeError(
            "vLLM full weight reload failed on all endpoints: " + "; ".join(errors)
        )

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

def export_actor_checkpoint(model, tokenizer, output_dir: Any) -> str:
    """Write a full HF checkpoint suitable for SGLang disk weight reload.

    Writes to ``*.partial`` then atomically renames so engines never load a
    half-written tree. For PEFT/LoRA, merges adapters for export only.
    """
    from pathlib import Path

    final = Path(output_dir)
    partial = final.parent / f"{final.name}.partial"
    if partial.exists():
        _rm_tree(partial)
    partial.mkdir(parents=True, exist_ok=True)
    unwrapped = getattr(model, "module", model)

    merged = False
    if hasattr(unwrapped, "merge_adapter"):
        unwrapped.merge_adapter()
        merged = True
    try:
        to_save = unwrapped
        if hasattr(unwrapped, "get_base_model"):
            to_save = unwrapped.get_base_model()
        to_save.save_pretrained(partial)
        tokenizer.save_pretrained(partial)
    finally:
        if merged and hasattr(unwrapped, "unmerge_adapter"):
            unwrapped.unmerge_adapter()
    if final.exists():
        _rm_tree(final)
    partial.rename(final)
    return str(final.resolve())


def export_actor_lora_adapter(model, tokenizer, output_dir: Any) -> str:
    """Write PEFT/LoRA adapter weights for vLLM dynamic ``load_lora_adapter``.

    Does **not** merge adapters. Atomic ``*.partial`` rename like full export.
    """
    from pathlib import Path

    final = Path(output_dir)
    partial = final.parent / f"{final.name}.partial"
    if partial.exists():
        _rm_tree(partial)
    partial.mkdir(parents=True, exist_ok=True)
    unwrapped = getattr(model, "module", model)
    # Prefer PEFT adapter export; fall back to full save for non-PEFT actors.
    if hasattr(unwrapped, "save_pretrained") and (
        hasattr(unwrapped, "peft_config")
        or hasattr(unwrapped, "get_base_model")
        or hasattr(unwrapped, "merge_adapter")
    ):
        unwrapped.save_pretrained(partial)
    else:
        unwrapped.save_pretrained(partial)
    tokenizer.save_pretrained(partial)
    if final.exists():
        _rm_tree(final)
    partial.rename(final)
    return str(final.resolve())


def _model_has_lora_adapters(model) -> bool:
    unwrapped = getattr(model, "module", model)
    return bool(
        hasattr(unwrapped, "peft_config")
        or hasattr(unwrapped, "get_base_model")
        or hasattr(unwrapped, "merge_adapter")
    )


def _resolve_vllm_weight_mode(config: SingleGpuSlimeConfig, model) -> str:
    mode = str(getattr(config, "vllm_weight_mode", "auto") or "auto").lower()
    if mode == "auto":
        if bool(getattr(config, "use_lora", False)) or _model_has_lora_adapters(model):
            return "lora"
        return "full"
    return mode

def _rm_tree(path: Any) -> None:
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        return
    for child in sorted(root.rglob("*"), reverse=True):
        with contextlib.suppress(OSError):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    with contextlib.suppress(OSError):
        root.rmdir()


def _actor_state_cpu(model) -> dict[str, Any]:
    """CPU snapshot of actor weights (merged view for PEFT when possible)."""
    unwrapped = getattr(model, "module", model)
    merged = False
    if hasattr(unwrapped, "merge_adapter"):
        unwrapped.merge_adapter()
        merged = True
    try:
        target = unwrapped
        if hasattr(unwrapped, "get_base_model"):
            target = unwrapped.get_base_model()
        state = {
            name: tensor.detach().float().cpu().clone()
            for name, tensor in target.state_dict().items()
        }
    finally:
        if merged and hasattr(unwrapped, "unmerge_adapter"):
            unwrapped.unmerge_adapter()
    return state


def _write_delta_payload(
    *,
    prev: dict[str, Any],
    curr: dict[str, Any],
    out_dir: Any,
    step: int,
) -> tuple[str, int, int]:
    """Write changed tensors as safetensors delta (slime-style disk delta)."""
    from pathlib import Path

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    changed: dict[str, Any] = {}
    for name, tensor in curr.items():
        old = prev.get(name)
        if old is None or old.shape != tensor.shape or not bool((old == tensor).all().item()):
            changed[name] = tensor
    meta = {
        "format": "seiso_sglang_delta_v1",
        "step": int(step),
        "num_tensors_total": len(curr),
        "num_tensors_changed": len(changed),
        "tensor_names": sorted(changed),
    }
    (path / "delta_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if changed:
        try:
            from safetensors.torch import save_file

            save_file(changed, str(path / "delta.safetensors"))
        except Exception:
            # Fallback without safetensors dependency shape
            import torch

            torch.save(changed, path / "delta.pt")
    return str(path.resolve()), len(changed), len(curr)


def _prune_weight_versions(weight_root: Any, keep: int) -> None:
    """Prune both full ``weight_v*`` and delta ``delta_v*`` version dirs."""
    from pathlib import Path

    root = Path(weight_root)
    if not root.is_dir() or keep < 1:
        return
    for prefix in ("weight_v", "delta_v"):
        versions = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
            key=lambda p: p.name,
        )
        for old in versions[:-keep]:
            _rm_tree(old)


class WeightSyncState:
    """Mutable rank0 state for full/delta SGLang weight transport."""

    def __init__(self) -> None:
        self.prev_state: dict[str, Any] | None = None
        self.last_full_path: str | None = None
        self.last_mode: str | None = None
        self.last_changed: int = 0


def _broadcast_weights_to_engines(
    config: SingleGpuSlimeConfig,
    *,
    model_path: str,
    weight_version: str,
) -> None:
    engines = sglang_engine_urls(config)
    if not engines:
        raise ValueError("sglang_base_url is required for SGLang weight sync")
    errors: list[str] = []
    for base in engines:
        client = SGLangRolloutClient.from_config(config)
        client.base_url = base
        try:
            client.update_weights_from_disk(model_path, weight_version=weight_version)
            client.flush_cache()
        except RuntimeError as exc:
            errors.append(f"{base}: {exc}")
    if errors:
        raise RuntimeError("SGLang weight sync failed on one or more engines: " + "; ".join(errors))


def sync_sglang_weights_from_actor(
    *,
    model,
    tokenizer,
    config: SingleGpuSlimeConfig,
    step: int,
    is_main: bool,
    active_backend: str | None = None,
    sync_state: WeightSyncState | None = None,
) -> str | None:
    """Rank-0 export + multi-engine SGLang hot-reload (slime disk transport).

    Modes (``sglang_weight_mode``):
    * ``full`` — always write a complete HF checkpoint + ``update_weights_from_disk``
    * ``delta`` — skip if no tensors changed; else try slime ``/pull_weights`` with a
      safetensors delta, falling back to full disk reload on vanilla SGLang

    Returns the checkpoint path when a sync ran on this process, else None.
    Non-main ranks return None (caller must barrier).
    """
    if not bool(getattr(config, "sglang_sync_weights", True)):
        return None
    backend = active_backend or resolve_rollout_backend(config, world_size=1)
    if backend != "sglang":
        return None
    if not is_main:
        return None

    mode = str(getattr(config, "sglang_weight_mode", "full") or "full").lower()
    if mode not in {"full", "delta"}:
        raise ValueError("sglang_weight_mode must be 'full' or 'delta'")

    weight_root = config.output_dir / str(
        getattr(config, "sglang_weight_dir", "sglang_weight_sync") or "sglang_weight_sync"
    )
    weight_root.mkdir(parents=True, exist_ok=True)
    version = f"v{int(step)}"
    ckpt_path = weight_root / f"weight_v{int(step):06d}"
    state = sync_state or WeightSyncState()
    keep = int(getattr(config, "sglang_weight_keep", 2) or 2)

    # ---- delta: snapshot only when needed (avoid full float clone on full mode) ----
    if mode == "delta" and state.prev_state is not None:
        curr = _actor_state_cpu(model)
        delta_dir = weight_root / f"delta_v{int(step):06d}"
        delta_path, n_changed, _n_total = _write_delta_payload(
            prev=state.prev_state,
            curr=curr,
            out_dir=delta_dir,
            step=step,
        )
        state.last_changed = n_changed
        if n_changed == 0:
            state.last_mode = "delta_skip"
            _prune_weight_versions(weight_root, keep)
            return state.last_full_path or delta_path

        # Try slime-patched /pull_weights on each engine (apply delta → local full).
        if state.last_full_path:
            pull_ok = True
            for base in sglang_engine_urls(config):
                client = SGLangRolloutClient.from_config(config)
                client.base_url = base
                try:
                    client.pull_weights(
                        local_checkpoint_dir=state.last_full_path,
                        source_dir=delta_path,
                        target_version=int(step),
                    )
                except RuntimeError:
                    pull_ok = False
                    break
            if pull_ok:
                for base in sglang_engine_urls(config):
                    client = SGLangRolloutClient.from_config(config)
                    client.base_url = base
                    client.flush_cache()
                state.prev_state = curr
                state.last_mode = "delta"
                _prune_weight_versions(weight_root, keep)
                return delta_path
        # Vanilla SGLang: fall through to full checkpoint.

    # ---- full HF checkpoint (default + delta fallback) ----
    used_path = export_actor_checkpoint(model, tokenizer, ckpt_path)
    _broadcast_weights_to_engines(config, model_path=used_path, weight_version=version)
    # Keep CPU snapshot only for future delta diffs (not for full-only runs).
    if mode == "delta":
        state.prev_state = _actor_state_cpu(model)
    else:
        state.prev_state = None
    state.last_full_path = used_path
    state.last_mode = "full"
    state.last_changed = 0
    _prune_weight_versions(weight_root, keep)
    return used_path


def _broadcast_vllm_lora(
    config: SingleGpuSlimeConfig,
    *,
    lora_path: str,
    lora_name: str,
) -> None:
    engines = vllm_engine_urls(config, allow_empty_primary=True)
    if not engines:
        base = resolve_vllm_base_url(config)
        engines = [base] if base else []
    if not engines:
        raise ValueError("vllm_base_url is required for vLLM weight sync")
    errors: list[str] = []
    for base in engines:
        client = VLLMRolloutClient.from_config(config)
        client.base_url = base
        try:
            client.load_lora_adapter(lora_path, lora_name=lora_name)
        except RuntimeError as exc:
            errors.append(f"{base}: {exc}")
    if errors:
        raise RuntimeError(
            "vLLM LoRA weight sync failed on one or more engines: " + "; ".join(errors)
        )


def _broadcast_vllm_full(
    config: SingleGpuSlimeConfig,
    *,
    model_path: str,
    weight_version: str,
) -> None:
    engines = vllm_engine_urls(config, allow_empty_primary=True)
    if not engines:
        base = resolve_vllm_base_url(config)
        engines = [base] if base else []
    if not engines:
        raise ValueError("vllm_base_url is required for vLLM weight sync")
    errors: list[str] = []
    for base in engines:
        client = VLLMRolloutClient.from_config(config)
        client.base_url = base
        try:
            client.pause()
            client.update_weights_from_disk(model_path, weight_version=weight_version)
            client.resume()
        except RuntimeError as exc:
            errors.append(f"{base}: {exc}")
    if errors:
        raise RuntimeError(
            "vLLM full weight sync failed on one or more engines: " + "; ".join(errors)
            + ". Prefer slime_use_lora + vllm_weight_mode=lora (dynamic "
            "/v1/load_lora_adapter), or restart the vLLM server from the exported "
            "checkpoint path."
        )


def sync_vllm_weights_from_actor(
    *,
    model,
    tokenizer,
    config: SingleGpuSlimeConfig,
    step: int,
    is_main: bool,
    active_backend: str | None = None,
    sync_state: WeightSyncState | None = None,
) -> str | None:
    """Rank-0 export + multi-engine vLLM hot-reload for multi-GPU slime rollouts.

    Modes (``vllm_weight_mode``):
    * ``auto`` — LoRA when the actor has PEFT adapters, else full
    * ``lora`` — export PEFT adapter + ``/v1/load_lora_adapter`` (preferred)
    * ``full`` — export merged HF checkpoint + best-effort disk reload endpoints

    Returns the checkpoint/adapter path when a sync ran on this process, else None.
    Non-main ranks return None (caller must barrier).
    """
    if not bool(getattr(config, "vllm_sync_weights", True)):
        return None
    backend = active_backend or resolve_rollout_backend(config, world_size=1)
    if backend != "vllm":
        return None
    if not is_main:
        return None

    mode = _resolve_vllm_weight_mode(config, model)
    weight_root = config.output_dir / str(
        getattr(config, "vllm_weight_dir", "vllm_weight_sync") or "vllm_weight_sync"
    )
    weight_root.mkdir(parents=True, exist_ok=True)
    version = f"v{int(step)}"
    keep = int(getattr(config, "vllm_weight_keep", 2) or 2)
    state = sync_state or WeightSyncState()
    lora_name = str(
        getattr(config, "vllm_lora_name", "") or "seiso_slime_policy"
    ).strip() or "seiso_slime_policy"

    if mode == "lora":
        adapter_path = weight_root / f"lora_v{int(step):06d}"
        used_path = export_actor_lora_adapter(model, tokenizer, adapter_path)
        _broadcast_vllm_lora(config, lora_path=used_path, lora_name=lora_name)
        state.last_full_path = used_path
        state.last_mode = "lora"
        state.last_changed = 0
        # Prune both lora_v* and weight_v* trees under the same root.
        _prune_weight_versions(weight_root, keep)
        _prune_named_versions(weight_root, prefix="lora_v", keep=keep)
        return used_path

    ckpt_path = weight_root / f"weight_v{int(step):06d}"
    used_path = export_actor_checkpoint(model, tokenizer, ckpt_path)
    _broadcast_vllm_full(config, model_path=used_path, weight_version=version)
    state.last_full_path = used_path
    state.last_mode = "full"
    state.last_changed = 0
    _prune_weight_versions(weight_root, keep)
    return used_path


def _prune_named_versions(weight_root: Any, *, prefix: str, keep: int) -> None:
    from pathlib import Path

    root = Path(weight_root)
    if not root.is_dir() or keep < 1:
        return
    versions = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
    )
    for old in versions[:-keep]:
        _rm_tree(old)