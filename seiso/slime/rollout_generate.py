"""Prompt formatting and online generation chunks for slime rollouts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.rollout_clients import (
    SGLangRolloutClient,
    VLLMRolloutClient,
)


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
    # OpenAI-style finish_reason per completion (HTTP backends); None for HF.
    finish_reasons: list[str | None] | None = None


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
        except Exception as exc:
            raise ValueError(
                "apply_chat_template failed for slime rollouts; fix the tokenizer "
                "chat template or set apply_chat_template=false. "
                f"({exc})"
            ) from exc
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
    # Held-out eval uses temperature=0 for greedy pass-rate reports. Sampling
    # at temp 0 is undefined across transformers versions — use do_sample=False.
    greedy = float(config.temperature) <= 0.0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "num_return_sequences": config.rollouts_per_prompt,
    }
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = config.temperature
        gen_kwargs["top_p"] = config.top_p
    with torch.no_grad():
        generated = generation_model.generate(**encoded, **gen_kwargs)
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


def truncate_prompt_texts(
    tokenizer,
    prompts: list[str],
    *,
    max_prompt_tokens: int,
) -> list[str]:
    """Token-truncate prompt strings to ``max_prompt_tokens``.

    HTTP rollout engines receive full text; actor logprobs in
    ``build_sequence_tensors`` truncate at encode time. Truncating here first
    keeps generate and importance-sampling prefixes aligned (same rule as the
    HF colocated path).
    """
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be >= 1")
    out: list[str] = []
    for prompt in prompts:
        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        ids = enc["input_ids"]
        # Batch vs single encodings depending on tokenizer wrapper.
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        out.append(tokenizer.decode(ids, skip_special_tokens=False))
    return out


def generate_sglang_chunk(
    *,
    tokenizer,
    prompts: list[str],
    config: SingleGpuSlimeConfig,
) -> GeneratedChunk:
    """Generate ``rollouts_per_prompt`` completions per prompt via SGLang HTTP."""
    from seiso.slime.rollout_http import sglang_engine_urls

    # Truncate before the engine sees the prompt so logprobs match.
    prompts = truncate_prompt_texts(tokenizer, prompts, max_prompt_tokens=config.max_prompt_tokens)
    client = SGLangRolloutClient.from_config(config)
    return _generate_http_chunk(
        client=client,
        prompts=prompts,
        rollouts_per_prompt=config.rollouts_per_prompt,
        max_workers=int(getattr(config, "sglang_max_workers", 8) or 8),
        engine_urls=sglang_engine_urls(config),
    )


def generate_vllm_chunk(
    *,
    tokenizer,
    prompts: list[str],
    config: SingleGpuSlimeConfig,
) -> GeneratedChunk:
    """Generate ``rollouts_per_prompt`` completions per prompt via vLLM HTTP."""
    from seiso.slime.rollout_http import resolve_vllm_base_url, vllm_engine_urls

    prompts = truncate_prompt_texts(tokenizer, prompts, max_prompt_tokens=config.max_prompt_tokens)
    client = VLLMRolloutClient.from_config(config)
    engines = vllm_engine_urls(config, allow_empty_primary=True)
    if not engines:
        base = resolve_vllm_base_url(config)
        engines = [base] if base else [client.base_url]
    return _generate_http_chunk(
        client=client,
        prompts=prompts,
        rollouts_per_prompt=config.rollouts_per_prompt,
        max_workers=int(getattr(config, "vllm_max_workers", 8) or 8),
        engine_urls=engines,
    )


def _generate_http_chunk(
    *,
    client: Any,
    prompts: list[str],
    rollouts_per_prompt: int,
    max_workers: int,
    engine_urls: list[str] | None = None,
) -> GeneratedChunk:
    """Shared OpenAI ``/v1/completions`` fan-out for SGLang and vLLM.

    When multiple engine URLs are configured, jobs are round-robined across
    engines (weight sync still fans out to all).
    """
    import copy

    n = int(rollouts_per_prompt)
    jobs: list[tuple[int, str]] = []
    for p_idx, prompt in enumerate(prompts):
        for _ in range(n):
            jobs.append((p_idx, prompt))

    engines = [u.rstrip("/") for u in (engine_urls or []) if str(u).strip()]
    if not engines:
        engines = [str(client.base_url).rstrip("/")]

    # One client per engine so concurrent workers never race on base_url.
    engine_clients: list[Any] = []
    for url in engines:
        cloned = copy.copy(client)
        cloned.base_url = url
        engine_clients.append(cloned)

    results: list[Any | None] = [None] * len(jobs)
    workers = min(max(1, int(max_workers)), max(1, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                engine_clients[idx % len(engine_clients)].complete_http,
                prompt,
            ): idx
            for idx, (_p_idx, prompt) in enumerate(jobs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    completions: list[str] = []
    token_id_lists: list[list[int] | None] = []
    finish_reasons: list[str | None] = []
    for item in results:
        if item is None:
            completions.append("")
            token_id_lists.append(None)
            finish_reasons.append(None)
        else:
            # HttpCompletion or legacy (text, token_ids) tuple from older mocks.
            if hasattr(item, "text"):
                completions.append(str(item.text))
                token_id_lists.append(getattr(item, "token_ids", None))
                finish_reasons.append(getattr(item, "finish_reason", None))
            else:
                text, tids = item
                completions.append(text)
                token_id_lists.append(tids)
                finish_reasons.append(None)
    expanded_prompts = [prompt for _p_idx, prompt in jobs]
    return GeneratedChunk(
        prompts=expanded_prompts,
        completions=completions,
        sequences=None,
        prompt_width=None,
        completion_token_ids=token_id_lists,
        finish_reasons=finish_reasons,
    )


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
