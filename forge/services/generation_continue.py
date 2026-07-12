"""Safe multi-pass continuation when a reply hits max_tokens (no n_ctx growth)."""

from __future__ import annotations

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


def total_reply_token_budget() -> int:
    """Max cumulative output tokens across all auto-continue passes."""
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


def max_auto_continues(*, pass_max_tokens: int | None = None) -> int:
    """Max extra generation passes after a length stop (0 disables).

    When ``SEISO_CHAT_AUTO_CONTINUE_MAX`` is unset (or set to -1), the count is
    derived so the total reply budget can be reached at the given per-pass size.
    Explicit non-negative env values are honored up to ``_HARD_MAX_CONTINUES``.
    """
    raw = env_int("SEISO_CHAT_AUTO_CONTINUE_MAX", _AUTO_CONTINUE_SENTINEL)
    if raw >= 0:
        return max(0, min(_HARD_MAX_CONTINUES, int(raw)))

    per = max(1, int(pass_max_tokens or 512))
    budget = total_reply_token_budget()
    # First generation + N continues must cover the full budget.
    needed_passes = max(1, (budget + per - 1) // per)
    continues = max(0, needed_passes - 1)
    return min(_HARD_MAX_CONTINUES, continues)


def hit_length_limit(
    output_tokens: int,
    max_tokens: int,
    *,
    finish_reason: str | None = None,
) -> bool:
    """True when generation stopped because the reply budget was exhausted."""
    reason = (finish_reason or "").strip().lower()
    if reason in {"length", "max_tokens"}:
        return True
    limit = max(1, int(max_tokens))
    # Allow ±1 for off-by-one token metering across backends.
    return int(output_tokens) >= max(1, limit - 1)


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
) -> bool:
    """Whether another short generation pass is warranted and safe to attempt."""
    if cancelled:
        return False
    if max_continues is None:
        max_continues = max_auto_continues(pass_max_tokens=max_tokens)
    if continues_used >= max(0, int(max_continues)):
        return False
    if total_budget is None:
        total_budget = total_reply_token_budget()
    # Leave room for at least a short next pass before spending another continue.
    if int(total_output_tokens) >= max(1, int(total_budget) - 8):
        return False
    if not str(pass_text or "").strip():
        return False
    # Avoid continue loops when the pass produced almost nothing useful.
    if int(pass_output_tokens) < 8 and (finish_reason or "").lower() not in {
        "length",
        "max_tokens",
    }:
        return False
    return hit_length_limit(
        pass_output_tokens, max_tokens, finish_reason=finish_reason
    )


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
