"""Chat context window stats for the UI."""

from __future__ import annotations

from typing import Any

from forge.services.chat_messages import (
    _context_char_budget,
    _message_cost,
    prepare_chat_context,
    trim_messages_to_context,
)
from forge.services.model_prompts import resolve_model_key
from seiso.inference.tuning import estimate_llama_n_ctx
from seiso.memory.protection import (
    _MAX_LLAMA_CTX,
    _MIN_LLAMA_CTX,
    _estimate_prompt_tokens,
    clamp_llama_n_ctx,
)


def _history_from_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": str(m["role"]),
            "content": str(m["content"]),
            "created_at": str(m.get("created_at") or ""),
        }
        for m in records
        if m.get("role") in ("user", "assistant")
    ]


def compute_chat_context_status(
    history: list[dict[str, str]],
    *,
    max_tokens: int = 2048,
    n_ctx: int | None = None,
    model_key: str = "default",
    tools_enabled: bool = False,
    knowledge_context: str | None = None,
) -> dict[str, Any]:
    """Return prompt budget, trim stats, and effective llama.cpp context size."""
    char_budget = _context_char_budget()
    raw_chars = sum(_message_cost(m) for m in history)
    trimmed = trim_messages_to_context(history)
    trimmed_chars = sum(_message_cost(m) for m in trimmed)
    prepared = prepare_chat_context(
        trimmed,
        model_key=model_key,
        tools_enabled=tools_enabled,
        knowledge_context=knowledge_context,
    )
    est_prompt_tokens = _estimate_prompt_tokens(prepared)
    n_ctx_auto = estimate_llama_n_ctx(prepared, max_tokens=max_tokens)
    n_ctx_max = clamp_llama_n_ctx(_MAX_LLAMA_CTX, messages=prepared, max_tokens=max_tokens)
    if n_ctx is None:
        n_ctx_effective = n_ctx_auto
    else:
        n_ctx_effective = clamp_llama_n_ctx(n_ctx, messages=prepared, max_tokens=max_tokens)

    reserved = max(1, int(max_tokens))
    total_need = est_prompt_tokens + reserved
    fill_ratio = min(1.0, total_need / max(n_ctx_effective, 1))

    return {
        "char_budget": char_budget,
        "char_used": trimmed_chars,
        "char_total": raw_chars,
        "message_count": len(history),
        "messages_included": len(trimmed),
        "messages_omitted": max(0, len(history) - len(trimmed)),
        "estimated_prompt_tokens": est_prompt_tokens,
        "max_tokens": reserved,
        "n_ctx": n_ctx_effective,
        "n_ctx_auto": n_ctx_auto,
        "n_ctx_min": _MIN_LLAMA_CTX,
        "n_ctx_max": n_ctx_max,
        "context_tokens_used": total_need,
        "context_tokens_limit": n_ctx_effective,
        "fill_ratio": round(fill_ratio, 4),
        "history_trimmed": len(trimmed) < len(history) or trimmed_chars < raw_chars,
    }


def context_status_for_history(
    history: list[dict[str, Any]],
    *,
    max_tokens: int = 2048,
    n_ctx: int | None = None,
    model_id: str | None = None,
    model_path: str | None = None,
    tools_enabled: bool = False,
    knowledge_context: str | None = None,
) -> dict[str, Any]:
    model_key = resolve_model_key(
        model_id=model_id,
        model_path=model_path,
    )
    return compute_chat_context_status(
        _history_from_records(history),
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        model_key=model_key,
        tools_enabled=tools_enabled,
        knowledge_context=knowledge_context,
    )
