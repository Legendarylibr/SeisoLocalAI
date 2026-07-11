"""Safe multi-pass continuation when a reply hits max_tokens (no n_ctx growth)."""

from __future__ import annotations

from typing import Any

from seiso.env import env_int

# Keep short so continue turns stay cheap and do not bloat the prompt.
CONTINUE_USER_PROMPT = (
    "Continue the previous assistant reply from exactly where it stopped. "
    "Do not restart, rephrase, or repeat earlier text. Do not add a preamble."
)

# Cap auto-continues: each pass reuses the loaded model/n_ctx (no KV resize).
_DEFAULT_MAX_CONTINUES = 2
_HARD_MAX_CONTINUES = 4


def max_auto_continues() -> int:
    """Max extra generation passes after a length stop (0 disables)."""
    return max(0, min(_HARD_MAX_CONTINUES, env_int("SEISO_CHAT_AUTO_CONTINUE_MAX", _DEFAULT_MAX_CONTINUES)))


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
) -> bool:
    """Whether another short generation pass is warranted and safe to attempt."""
    if cancelled:
        return False
    if max_continues is None:
        max_continues = max_auto_continues()
    if continues_used >= max(0, int(max_continues)):
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
) -> list[dict[str, Any]]:
    """Messages for a continuation pass: history + partial assistant + continue cue."""
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
