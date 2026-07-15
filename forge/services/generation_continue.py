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
# Used after an empty continue pass or a clear mid-sentence cut.
CONTINUE_USER_PROMPT_STRONG = (
    "Your previous assistant message was cut off mid-sentence. "
    "Continue EXACTLY from the final character with no restart, no new title, "
    "and no preamble. Finish all remaining sections completely."
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
    # Follow-ups that ask to extend / redo the previous long reply.
    re.compile(
        r"^\s*("
        r"longer|longer\s+please|make\s+it\s+longer|more|more\s+please|"
        r"continue|keep\s+going|expand|extend|again|once\s+more|one\s+more|"
        r"retry|rewrite|another|go\s+on|finish\s+it|complete\s+it"
        r")\s*[.!]?\s*$",
        re.I,
    ),
    re.compile(r"\b(write\s+a\s+song|song\s+lyrics)\b", re.I),
)

# Strong end-of-reply punctuation (last line).
_LAST_LINE_DONE_RE = re.compile(r'[.!?…]\s*[\"\'»”’)\]\*]*\s*$')
# Mid-thought endings that should always trigger another pass.
_INCOMPLETE_TRAIL_RE = re.compile(
    r"(?:"
    r"[,;:—–\-]\s*$|"  # mid-clause punctuation
    r"\b(the|a|an|and|or|but|to|of|in|on|for|with|as|at|by|from|into|"
    r"than|that|which|who|i'm|i|we|they|you|not|just|so|let|explore)\s*$|"
    r"\*{0,2}\([A-Za-z][^)\n]{0,40}$|"  # broken section marker *(Coda / *(Outro
    r"—\s*$"
    r")",
    re.I,
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
    if not text or not text.strip():
        return False
    sample = text[:4000]
    # Short follow-ups ("again", "more") are intentional long-form signals.
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


def _last_nonempty_line(text: str) -> str:
    for line in reversed(str(text or "").splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _has_broken_markup(text: str) -> bool:
    body = str(text or "")
    if body.count("*(") > body.count(")*"):
        return True
    if body.count("**") % 2 == 1:
        return True
    # Unclosed italic section headers like *(Coda without closing.
    return bool(re.search(r"\*\([A-Za-z][^)\n]{0,40}$", body.rstrip()))


def looks_incomplete_reply(text: str) -> bool:
    """Heuristic: model stopped mid-thought even though finish_reason was stop.

    Live chats (songs / “again” / “longer”) often end mid-line with EOS while
    still under the per-pass token cap — length-based auto-continue never fires.
    """
    body = str(text or "").rstrip()
    if len(body) < 24:
        return False
    last_line = _last_nonempty_line(body)
    if not last_line:
        return False

    if _has_broken_markup(body) or _has_broken_markup(last_line):
        return True
    if _INCOMPLETE_TRAIL_RE.search(last_line):
        return True
    # Clean finish: last line ends with real sentence punctuation.
    if _LAST_LINE_DONE_RE.search(last_line):
        return False
    # Otherwise treat as mid-thought (word / clause cut-off).
    return last_line[-1].isalnum() or last_line[-1] in {
        "*",
        "_",
        "—",
        "–",
        "-",
        ",",
        ";",
        ":",
        "(",
        "[",
        "{",
    }


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
    force_incomplete: bool = False,
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
    text = str(pass_text or "").strip()
    if not text and not force_incomplete:
        return False

    tokens = effective_pass_tokens(
        pass_output_tokens, pass_text=pass_text, metadata=metadata
    )
    # Avoid continue loops when the pass produced almost nothing useful.
    reason = (finish_reason or "").lower()
    if tokens < 8 and reason not in {"length", "max_tokens"} and not force_incomplete:
        return False

    if hit_length_limit(
        tokens,
        max_tokens,
        finish_reason=finish_reason,
        pass_text=pass_text,
        metadata=metadata,
    ):
        return True

    # Model emitted stop/EOS mid-sentence under the per-pass cap (common on
    # small instruct models for songs/essays). Keep going while budget remains.
    if force_incomplete:
        return True
    # Low bar: any clear incomplete draft past a short stub.
    return tokens >= 24 and looks_incomplete_reply(pass_text)


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
    """True when the reply is incomplete and no further continue is possible.

    Covers both hard length-stops and mid-sentence EOS under the pass cap.
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
    incomplete = looks_incomplete_reply(pass_text)
    if not hit and not incomplete:
        return False
    # Incomplete/length-hit only surfaces as truncated if we cannot continue.
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


# Marker when we drop the middle of a long in-progress assistant draft.
_ASSISTANT_DECAY_MARKER = "\n[...earlier part of this reply omitted...]\n"
# Never pack the fixed n_ctx to the brim — leave free headroom for prefill.
_CONTINUE_MIN_FREE_RATIO = 0.12
_CONTINUE_MAX_FREE_RATIO = 0.30


def _text_token_estimate(text: str) -> int:
    return max(0, estimate_tokens_from_text(text))


def _keep_text_tail(text: str, token_budget: int) -> str:
    """Keep the most recent portion of *text* within *token_budget* tokens."""
    body = str(text or "")
    if token_budget <= 0 or not body:
        return ""
    if _text_token_estimate(body) <= token_budget:
        return body
    # char≈3.2/token matches chat_guards; keep the tail.
    char_budget = max(1, int(token_budget * 3.2))
    if len(body) <= char_budget:
        return body
    return body[-char_budget:]


def _keep_text_head(text: str, token_budget: int) -> str:
    body = str(text or "")
    if token_budget <= 0 or not body:
        return ""
    if _text_token_estimate(body) <= token_budget:
        return body
    char_budget = max(1, int(token_budget * 3.2))
    if len(body) <= char_budget:
        return body
    return body[:char_budget]


def linear_decay_fill_ratio(*, assistant_tokens: int, n_ctx: int) -> float:
    """How full the prompt window may be (rest is free headroom).

    Linear decay: as the in-progress reply grows relative to ``n_ctx``, pack
    *less* of the window so prefill never sits at the context edge (OOM /
    quality cliff). Starts ~88% full, decays toward ~70% full.
    """
    ctx = max(1, int(n_ctx))
    ratio = min(1.0, max(0.0, float(assistant_tokens) / float(ctx)))
    # fill = 0.88 - 0.18 * (assistant/n_ctx)
    fill = 0.88 - 0.18 * ratio
    return max(1.0 - _CONTINUE_MAX_FREE_RATIO, min(1.0 - _CONTINUE_MIN_FREE_RATIO, fill))


def decay_assistant_draft(text: str, token_budget: int) -> str:
    """Keep head + recent tail of the draft; drop the middle with linear head decay.

    Older middle content is discarded first so the model always sees the latest
    sentences (where generation stopped) plus a shrinking topic opener.
    """
    body = str(text or "")
    budget = max(0, int(token_budget))
    if budget <= 0 or not body.strip():
        return ""
    total = _text_token_estimate(body)
    if total <= budget:
        return body

    # Overflow ratio in [0, 1]: how much we exceed the budget.
    overflow = min(1.0, (total - budget) / max(1.0, float(total)))
    # Head share decays linearly from ~30% → ~8% as overflow grows.
    head_frac = max(0.08, 0.30 - 0.22 * overflow)
    head_budget = max(24, int(budget * head_frac))
    tail_budget = max(48, budget - head_budget - _text_token_estimate(_ASSISTANT_DECAY_MARKER))
    if tail_budget < 32:
        # Extreme pressure: recent tail only.
        return _keep_text_tail(body, budget)

    head = _keep_text_head(body, head_budget)
    tail = _keep_text_tail(body, tail_budget)
    if not head:
        return tail
    if not tail:
        return head
    # Avoid duplicating when head/tail overlap on short strings.
    if tail in head or head in tail:
        return _keep_text_tail(body, budget)
    return f"{head}{_ASSISTANT_DECAY_MARKER}{tail}"


def pack_continue_messages_linear_decay(
    messages: list[dict[str, Any]],
    *,
    n_ctx: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Fit continue turns into fixed ``n_ctx`` with free headroom + recent-first decay.

    Priority (high → low):
      1. continue cue (last user)
      2. recent tail of in-progress assistant draft
      3. latest original user task
      4. system messages
      5. older history (dropped first)
    """
    if not messages:
        return []

    ctx = max(1, int(n_ctx))
    gen = max(1, int(max_tokens))
    # Hard ceiling for prompt tokens (same reserve idea as chat_guards).
    hard_prompt_budget = max(256, ctx - gen - 128)

    # Locate the in-progress assistant draft (second-to-last when structure is
    # history + assistant + continue cue).
    assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if str(messages[i].get("role") or "").lower() == "assistant":
            assistant_idx = i
            break
    assistant_text = ""
    if assistant_idx is not None:
        assistant_text = str(messages[assistant_idx].get("content") or "")
    assist_tokens = _text_token_estimate(assistant_text)

    fill = linear_decay_fill_ratio(assistant_tokens=assist_tokens, n_ctx=ctx)
    prompt_budget = max(128, int(hard_prompt_budget * fill))

    packed = [dict(m) for m in messages]

    # 1) Always keep the continue cue short and intact.
    if packed and str(packed[-1].get("role") or "").lower() == "user":
        cue = str(packed[-1].get("content") or CONTINUE_USER_PROMPT)
        if _text_token_estimate(cue) > 64:
            packed[-1]["content"] = _keep_text_tail(cue, 64)

    # 2) Linear-decay the growing assistant draft (head shrinks, tail preserved).
    if assistant_idx is not None:
        # Give the draft the majority of the prompt window; leave room for task/system.
        draft_share = max(96, int(prompt_budget * 0.58))
        packed[assistant_idx]["content"] = decay_assistant_draft(
            str(packed[assistant_idx].get("content") or ""),
            draft_share,
        )

    # 3) Drop oldest conversational turns until under budget (keep system + last cue).
    def _est(msgs: list[dict[str, Any]]) -> int:
        total = 0
        for m in msgs:
            total += max(1, _text_token_estimate(str(m.get("content") or "")))
        return max(64, total)

    # Indices that are safe to drop: user/assistant before the draft (not system).
    drop_idx = 0
    last_keep = len(packed) - 1  # continue cue
    while _est(packed) > prompt_budget and drop_idx < last_keep:
        role = str(packed[drop_idx].get("role") or "").lower()
        # Never drop the in-progress assistant draft or the continue cue.
        if drop_idx in (assistant_idx, last_keep):
            drop_idx += 1
            continue
        if role in {"user", "assistant", "tool"}:
            packed.pop(drop_idx)
            if assistant_idx is not None and drop_idx < assistant_idx:
                assistant_idx -= 1
            last_keep = len(packed) - 1
            continue
        drop_idx += 1

    # 4) If still over budget, shrink system / remaining history from the front,
    #    then further compress the assistant draft (tail-heavy).
    guard = 0
    while _est(packed) > prompt_budget and guard < 32:
        guard += 1
        overflow = _est(packed) - prompt_budget
        # Prefer shrinking the longest non-cue message that isn't pure system-first.
        candidates = []
        for i, m in enumerate(packed):
            if i == last_keep:
                continue
            content = str(m.get("content") or "")
            tok = _text_token_estimate(content)
            if tok <= 32:
                continue
            role = str(m.get("role") or "").lower()
            # Prefer trimming older / longer content; protect draft tail via decay.
            priority = tok
            if role == "system":
                priority -= 50
            if i == assistant_idx:
                priority += 100  # trim draft via decay helper, not last
            candidates.append((priority, i, tok))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, idx, tok = candidates[0]
        target = max(32, tok - overflow - 8)
        if idx == assistant_idx:
            packed[idx]["content"] = decay_assistant_draft(
                str(packed[idx].get("content") or ""),
                target,
            )
        else:
            packed[idx]["content"] = _keep_text_tail(str(packed[idx].get("content") or ""), target)

    # 5) Final safety: if still over hard budget, force tail-only draft + cue + one user.
    if _est(packed) > hard_prompt_budget:
        cue_content = CONTINUE_USER_PROMPT
        if packed and str(packed[-1].get("role") or "").lower() == "user":
            cue_content = str(packed[-1].get("content") or CONTINUE_USER_PROMPT)
        draft = ""
        if assistant_idx is not None and 0 <= assistant_idx < len(packed):
            draft = str(packed[assistant_idx].get("content") or "")
        # Recover original user task if present (not the continue cue).
        user_task = ""
        last_i = len(packed) - 1
        for i, m in enumerate(packed):
            if i == last_i:
                continue
            if str(m.get("role") or "").lower() == "user":
                user_task = str(m.get("content") or "")
                break
        room = max(128, hard_prompt_budget - 80)
        draft_room = max(64, int(room * 0.7))
        task_room = max(32, room - draft_room)
        rebuilt: list[dict[str, Any]] = []
        if user_task:
            rebuilt.append({"role": "user", "content": _keep_text_tail(user_task, task_room)})
        rebuilt.append(
            {
                "role": "assistant",
                "content": decay_assistant_draft(draft or assistant_text, draft_room),
            }
        )
        rebuilt.append({"role": "user", "content": cue_content})
        packed = rebuilt

    return packed


def build_continue_messages(
    base_messages: list[dict[str, Any]],
    assistant_so_far: str,
    *,
    n_ctx: int | None = None,
    max_tokens: int | None = None,
    strong: bool = False,
) -> list[dict[str, Any]]:
    """Messages for a continuation pass: history + partial assistant + continue cue.

    When ``n_ctx`` is provided, pack with **linear decay**: free headroom grows as
    the draft grows, older middle content is dropped first, and the recent tail
    of the in-progress reply is always preserved. Fixed ``n_ctx`` is never grown
    (OOM-safe multi-pass).

    ``strong=True`` uses a firmer continue cue after empty/incomplete cuts.
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
    cue = CONTINUE_USER_PROMPT_STRONG if strong else CONTINUE_USER_PROMPT
    messages.append({"role": "user", "content": cue})
    if n_ctx is not None:
        reply_budget = max(1, int(max_tokens or 512))
        messages = pack_continue_messages_linear_decay(
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
