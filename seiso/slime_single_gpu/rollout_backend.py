"""Online rollout backends for slime-style GRPO.

* ``data_gen`` — colocated Hugging Face ``generate`` (default; single-GPU path)
* ``sglang`` — OpenAI-compatible HTTP generation against a running SGLang server
  (preferred for multi-GPU / high-throughput generation)

Completions are always produced online. Prompt corpora (labels/tests) come from
JSONL or high-level ``data_gen`` corpus materialization — never from stored
model outputs.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from seiso.slime_single_gpu.config import SingleGpuSlimeConfig

# Canonical: hf | sglang | auto. "data_gen" is a legacy alias of "hf".
_ROLLOUT_BACKENDS = frozenset({"hf", "sglang", "auto", "data_gen"})


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
    * ``auto`` — SGLang when ``sglang_base_url`` is set and ``world_size > 1``
    """
    name = _normalize_backend_name(
        getattr(config, "rollout_backend", "hf") or "hf"
    )
    if name not in {"hf", "sglang", "auto"}:
        raise ValueError(
            f"rollout_backend must be one of: hf, sglang, auto (got {name!r})"
        )
    if name == "auto":
        base = str(getattr(config, "sglang_base_url", "") or "").strip()
        if base and world_size > 1:
            return "sglang"
        return "hf"
    return name


def validate_rollout_backend_config(config: SingleGpuSlimeConfig) -> None:
    name = _normalize_backend_name(
        getattr(config, "rollout_backend", "hf") or "hf"
    )
    if name not in {"hf", "sglang", "auto"}:
        raise ValueError(
            f"rollout_backend must be one of: hf, sglang, auto (got {name!r})"
        )
    if name == "sglang":
        base = str(getattr(config, "sglang_base_url", "") or "").strip()
        if not base:
            raise ValueError(
                "rollout_backend=sglang requires sglang_base_url "
                "(e.g. http://127.0.0.1:30000)"
            )
    timeout = float(getattr(config, "sglang_timeout_s", 120.0) or 120.0)
    if timeout <= 0:
        raise ValueError("sglang_timeout_s must be positive")
    max_workers = int(getattr(config, "sglang_max_workers", 8) or 8)
    if max_workers < 1:
        raise ValueError("sglang_max_workers must be positive")


@dataclass(frozen=True)
class GeneratedChunk:
    """One generation chunk: parallel lists aligned as prompt_idx * n + k."""

    prompts: list[str]
    completions: list[str]
    # When set (data_gen), full sequences from HF generate; else None → re-tokenize.
    sequences: Any | None = None
    prompt_width: int | None = None


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
                "content": (
                    f"{content.rstrip()}\n\n{config.thinking_instruction}\n<think>"
                ),
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
    n = config.rollouts_per_prompt
    client = SGLangRolloutClient.from_config(config)
    # Expand to one request per (prompt, sample) for broad API compatibility.
    jobs: list[tuple[int, str]] = []
    for p_idx, prompt in enumerate(prompts):
        for _ in range(n):
            jobs.append((p_idx, prompt))

    results: list[str | None] = [None] * len(jobs)
    max_workers = min(
        int(getattr(config, "sglang_max_workers", 8) or 8),
        max(1, len(jobs)),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(client.complete, prompt): idx
            for idx, (_p_idx, prompt) in enumerate(jobs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    completions = [text if text is not None else "" for text in results]
    expanded_prompts = [prompt for _p_idx, prompt in jobs]
    return GeneratedChunk(
        prompts=expanded_prompts,
        completions=completions,
        sequences=None,
        prompt_width=None,
    )


def build_sequence_tensors(
    *,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    config: SingleGpuSlimeConfig,
    torch,
    device: str,
) -> list[dict[str, Any]]:
    """Tokenize prompt+completion pairs into per-row rollout tensors."""
    rows: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for prompt, completion in zip(prompts, completions, strict=True):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=config.max_prompt_tokens,
            return_tensors="pt",
        )["input_ids"][0]
        # Completions are model text; do not add special tokens.
        if completion:
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
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        prompt_len = int(prompt_ids.numel())
        response_mask[prompt_len:] = True
        # Drop pad positions in the response (should be none without padding).
        if pad_id is not None:
            response_mask = response_mask & (input_ids != pad_id)
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
        base = str(getattr(config, "sglang_base_url", "") or "").strip()
        if not base:
            raise ValueError("sglang_base_url is required for SGLang rollout")
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
            # Some servers put chat-style content here.
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            raise RuntimeError("SGLang choice missing text")
        return text

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
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
            raise RuntimeError(f"SGLang HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"SGLang request failed: {exc}") from exc
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("SGLang returned non-object JSON")
        return parsed
