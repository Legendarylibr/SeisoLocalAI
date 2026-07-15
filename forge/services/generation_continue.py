"""Safe multi-pass continuation when a reply hits max_tokens (no n_ctx growth).

Per-pass ``max_tokens`` stays OOM-clamped by memory guards (often 512–768 on
native Linux NVIDIA). Long replies (research, papers, songs, etc.) finish by
running more fixed-size generation chunks under a dynamic total budget — never
by raising the single-pass completion size into OOM territory.
"""

from __future__ import annotations

import os
import re
from typing import Any

from seiso.env import env_int

# Keep short so continue turns stay cheap and do not bloat the prompt.
CONTINUE_USER_PROMPT = (
    "Continue the previous assistant reply from exactly where it stopped. "
    "Do not restart, rephrase, or repeat earlier text. Do not add a preamble."
)

# OOM safety is *not* enforced by limiting how many times we continue.
# It comes from:
#   - fixed n_ctx across passes (no KV growth)
#   - trim_llama_messages_to_context on the continue prompt
#   - per-pass max_tokens staying at the first-pass OOM-safe cap (often 512–768
#     on native Linux NVIDIA)
#
# These continue/total ceilings only guard against runaway generation loops.
# Default total is high enough that long essays finish without a mid-reply stop.
_DEFAULT_TOTAL_REPLY_TOKENS = 32768
# Host RAM for ~100k tokens of text is still tiny vs model VRAM; hard ceiling
# is runaway protection only.
_HARD_MAX_TOTAL_REPLY_TOKENS = 131072
# Absolute pass count ceiling when env forces a value or budget is huge.
_HARD_MAX_CONTINUES = 256
# When SEISO_CHAT_AUTO_CONTINUE_MAX is unset, derive pass count from the total
# budget and the OOM-safe per-pass size so small native caps never starve long
# replies (e.g. 32768 / 512 ≈ 64 continues).
_AUTO_CONTINUE_SENTINEL = -1
# Long-form prompts (paper / song / research) get at least this total when
# free memory allows — still delivered in OOM-safe chunks.
_LONG_FORM_TOTAL_FLOOR = 65536

_LONG_FORM_PATTERNS = (
    re.compile(r"\b(research\s+paper|white\s*paper|thesis|dissertation)\b", re.I),
    re.compile(
        r"\b(write|draft|compose|generate)\b.{0,40}\b(paper|essay|article|report)\b",
        re.I,
    ),
    re.compile(
        r"\b(write|draft|compose|generate)\b.{0,40}\b(song|lyrics|poem|screenplay|script)\b",
        re.I,
    ),
    re.compile(r"\b(full\s+song|song\s+lyrics|chapter|novel|long[- ]form)\b", re.I),
    re.compile(
        r"\b(in[- ]depth|comprehensive)\b.{0,40}\b(analysis|guide|review|report)\b",
        re.I,
    ),
    re.compile(r"\b(book\s+chapter|literature\s+review|technical\s+spec)\b", re.I),
)


def _env_int_override(name: str) -> int | None:
    """Return an explicit env int when set; ``None`` when unset/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def total_reply_token_budget() -> int:
    """Max cumulative output tokens across all auto-continue passes (env default)."""
    return max(
        256,
        min(
            _HARD_MAX_TOTAL_REPLY_TOKENS,
            env_int(
                "SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS",
                _DEFAULT_TOTAL_REPLY_TOKENS,
            ),
        ),
    )


def max_auto_continues(
    *,
    pass_max_tokens: int | None = None,
    total_budget: int | None = None,
) -> int:
    """Max extra generation passes after a length stop (0 disables).

    When ``SEISO_CHAT_AUTO_CONTINUE_MAX`` is unset (or set to -1), the count is
    derived so the total reply budget can be reached at the given per-pass size.
    Explicit non-negative env values are honored up to ``_HARD_MAX_CONTINUES``.
    """
    raw = env_int("SEISO_CHAT_AUTO_CONTINUE_MAX", _AUTO_CONTINUE_SENTINEL)
    if raw >= 0:
        return max(0, min(_HARD_MAX_CONTINUES, int(raw)))

    per = max(1, int(pass_max_tokens or 512))
    budget = (
        max(256, min(_HARD_MAX_TOTAL_REPLY_TOKENS, int(total_budget)))
        if total_budget is not None
        else total_reply_token_budget()
    )
    # First generation + N continues must cover the full budget.
    needed_passes = max(1, (budget + per - 1) // per)
    continues = max(0, needed_passes - 1)
    return min(_HARD_MAX_CONTINUES, continues)


def looks_long_form(messages: list[dict[str, Any]] | None) -> bool:
    """True when the latest user turn looks like a long-form generation ask."""
    if not messages:
        return False
    text = ""
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").lower() != "user":
            continue
        content = item.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            text = " ".join(parts)
        else:
            text = str(content or "")
        break
    if not text or len(text) < 12:
        return False
    sample = text[:4000]
    return any(pat.search(sample) for pat in _LONG_FORM_PATTERNS)


def resolve_auto_continue_limits(
    *,
    requested_max_tokens: int | None,
    pass_max_tokens: int,
    messages: list[dict[str, Any]] | None = None,
    headroom_mb: float | int | None = None,
) -> tuple[int, int]:
    """Compute ``(max_continues, total_token_budget)`` for multi-pass chat.

    * ``pass_max_tokens`` — OOM-safe per-pass clamp (already sanitized).
    * ``requested_max_tokens`` — client desire *before* per-pass clamp; used as
      a floor for cumulative output so short chunk size does not truncate papers.
    * Continues only fire on length stops; natural ``stop`` still ends early.
    """
    env_total = _env_int_override("SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS")
    pass_sz = max(1, int(pass_max_tokens or 1))
    requested = max(0, int(requested_max_tokens or 0))

    if env_total is not None:
        total = max(256, min(_HARD_MAX_TOTAL_REPLY_TOKENS, env_total))
    else:
        total = _DEFAULT_TOTAL_REPLY_TOKENS
        # Client max_tokens is the desired overall reply length signal.
        if requested > 0:
            total = max(total, min(_HARD_MAX_TOTAL_REPLY_TOKENS, requested))
        if looks_long_form(messages):
            total = max(total, min(_HARD_MAX_TOTAL_REPLY_TOKENS, _LONG_FORM_TOTAL_FLOOR))
            if requested > 0:
                total = max(total, min(_HARD_MAX_TOTAL_REPLY_TOKENS, requested))

        # Headroom only *caps* how far we inflate — never forces long rambles.
        if headroom_mb is not None:
            try:
                free = float(headroom_mb)
            except (TypeError, ValueError):
                free = 0.0
            if free < 1536:
                total = min(total, max(requested or 0, 4096, pass_sz * 2))
            elif free < 3072:
                total = min(total, max(requested or 0, 8192, pass_sz * 4))
            elif free < 6144:
                total = min(total, max(requested or 0, 16384, _DEFAULT_TOTAL_REPLY_TOKENS))

        total = max(256, min(_HARD_MAX_TOTAL_REPLY_TOKENS, total))

    max_continues = max_auto_continues(
        pass_max_tokens=pass_sz,
        total_budget=total,
    )
    return max_continues, total


def estimate_tokens_from_text(text: str) -> int:
    """Rough token floor from text when backends under-count stream tokens."""
    body = str(text or "").strip()
    if not body:
        return 0
    # English-ish: ~4 chars/token, ~1.3 tokens/word — take the higher floor.
    by_chars = max(1, len(body) // 4)
    words = body.split()
    by_words = max(1, int(len(words) * 1.3)) if words else 1
    return max(by_chars, by_words)


def effective_pass_tokens(
    metered_tokens: int,
    *,
    pass_text: str = "",
    metadata: dict[str, Any] | None = None,
) -> int:
    """Best available token count for length-stop decisions."""
    tokens = authoritative_pass_tokens(metered_tokens, metadata)
    return max(tokens, estimate_tokens_from_text(pass_text))


def hit_length_limit(
    output_tokens: int,
    max_tokens: int,
    *,
    finish_reason: str | None = None,
    pass_text: str = "",
    metadata: dict[str, Any] | None = None,
) -> bool:
    """True when this pass stopped because the per-pass completion budget ran out.

    Uses metered tokens, backend eval counts, and a text estimate so under-counted
    streams still trigger auto-continue instead of a false early stop.
    """
    limit = max(1, int(max_tokens))
    tokens = effective_pass_tokens(
        output_tokens, pass_text=pass_text, metadata=metadata
    )
    # Allow ±1 for off-by-one token metering across backends.
    near_cap = tokens >= max(1, limit - 1)
    # "Most of the pass" — catches slight under-counts without treating short
    # natural answers as length stops.
    substantial = tokens >= max(16, (limit * 2) // 3)

    reason = (finish_reason or "").strip().lower()
    if reason in {"length", "max_tokens"}:
        # Trust explicit length stops when the pass produced real content.
        # (Spurious empty length frames are filtered by should_auto_continue.)
        return near_cap or substantial or tokens >= 8
    if reason in {"stop", "eos", "end", "end_turn", "completed"}:
        # Some runtimes mislabel num_predict hits as stop — trust the meter.
        return near_cap
    return near_cap


def can_schedule_another_continue(
    *,
    continues_used: int,
    max_continues: int | None = None,
    total_output_tokens: int = 0,
    total_budget: int | None = None,
    pass_max_tokens: int | None = None,
) -> bool:
    """True when multi-pass budget still allows another OOM-safe chunk."""
    if max_continues is None:
        max_continues = max_auto_continues(pass_max_tokens=pass_max_tokens)
    if continues_used >= max(0, int(max_continues)):
        return False
    if total_budget is None:
        total_budget = total_reply_token_budget()
    # Need room for a meaningful next chunk (not a 1-token stub).
    remaining = max(0, int(total_budget) - int(total_output_tokens))
    return remaining >= 8


def should_auto_continue(
    *,
    pass_output_tokens: int,
    max_tokens: int,
    pass_text: str,
    continues_used: int,
    max_continues: int | None = None,
    finish_reason: str | None = None,
    cancelled: bool = False,
    total_output_tokens: int = 0,
    total_budget: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Whether another short generation pass is warranted and safe to attempt."""
    if cancelled:
        return False
    if not can_schedule_another_continue(
        continues_used=continues_used,
        max_continues=max_continues,
        total_output_tokens=total_output_tokens,
        total_budget=total_budget,
        pass_max_tokens=max_tokens,
    ):
        return False
    if not str(pass_text or "").strip():
        return False

    tokens = effective_pass_tokens(
        pass_output_tokens, pass_text=pass_text, metadata=metadata
    )
    # Avoid continue loops when the pass produced almost nothing useful.
    reason = (finish_reason or "").lower()
    if tokens < 8 and reason not in {"length", "max_tokens"}:
        return False

    return hit_length_limit(
        tokens,
        max_tokens,
        finish_reason=finish_reason,
        pass_text=pass_text,
        metadata=metadata,
    )


def reply_still_truncated(
    *,
    last_pass_tokens: int,
    pass_max_tokens: int,
    finish_reason: str | None,
    total_output_tokens: int,
    total_budget: int,
    continues_used: int,
    max_continues: int,
    pass_text: str = "",
    metadata: dict[str, Any] | None = None,
    cancelled: bool = False,
) -> bool:
    """True only when the last pass hit length *and* no further continue is possible.

    Avoids the false UI banner when a single OOM-safe chunk ends but multi-pass
    budget remains (or when the model naturally stopped mid-way).
    """
    if cancelled:
        return False
    hit = hit_length_limit(
        last_pass_tokens,
        pass_max_tokens,
        finish_reason=finish_reason,
        pass_text=pass_text,
        metadata=metadata,
    )
    if not hit:
        return False
    # Length-hit is only "truncated" if we could not schedule another chunk.
    return not can_schedule_another_continue(
        continues_used=continues_used,
        max_continues=max_continues,
        total_output_tokens=total_output_tokens,
        total_budget=total_budget,
        pass_max_tokens=pass_max_tokens,
    )


def next_pass_max_tokens(
    *,
    base_pass_max_tokens: int,
    total_output_tokens: int,
    total_budget: int,
) -> int:
    """OOM-safe per-pass size, also clamped to remaining multi-pass budget."""
    base = max(1, int(base_pass_max_tokens))
    remaining = max(0, int(total_budget) - int(total_output_tokens))
    if remaining <= 0:
        return 0
    return max(1, min(base, remaining))


def build_continue_messages(
    base_messages: list[dict[str, Any]],
    assistant_so_far: str,
    *,
    n_ctx: int | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Messages for a continuation pass: history + partial assistant + continue cue.

    When ``n_ctx`` is provided, trim to that fixed window so multi-pass continues
    never grow KV beyond the original load (OOM-safe).
    """
    messages: list[dict[str, Any]] = []
    for item in base_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = item.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        messages.append({"role": role, "content": "" if content is None else str(content)})
    messages.append({"role": "assistant", "content": str(assistant_so_far)})
    messages.append({"role": "user", "content": CONTINUE_USER_PROMPT})
    if n_ctx is not None:
        from seiso.memory.protection.chat_guards import trim_llama_messages_to_context

        reply_budget = max(1, int(max_tokens or 512))
        messages = trim_llama_messages_to_context(
            messages,
            n_ctx=max(1, int(n_ctx)),
            max_tokens=reply_budget,
        )
    return messages


def resolve_finish_reason(
    *,
    hit_length: bool,
    cancelled: bool = False,
    explicit: str | None = None,
) -> str:
    if cancelled:
        return "cancelled"
    if explicit:
        return explicit
    return "length" if hit_length else "stop"


def authoritative_pass_tokens(
    metered_tokens: int,
    metadata: dict[str, Any] | None,
) -> int:
    """Prefer backend-reported eval/completion counts over stream estimates."""
    tokens = max(0, int(metered_tokens))
    if not metadata:
        return tokens
    for key in ("eval_count", "completion_tokens", "output_tokens"):
        raw = metadata.get(key)
        if raw is None:
            continue
        try:
            reported = int(raw)
        except (TypeError, ValueError):
            continue
        if reported > 0:
            tokens = max(tokens, reported)
    return tokens
