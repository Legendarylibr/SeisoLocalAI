"""Unified thinking / internal-reasoning budget for every chat backend.

OOM-safe by design:
  * never grows ``n_ctx``
  * thinking is a small *extra* decode allotment, capped hard
  * content always keeps the majority of the pass budget
  * mid-stream thinking caps abort the pass so auto-continue can finish
    with a content-only retry

Quality-first defaults (``SEISO_THINK_MODE=auto``):
  * enable brief thinking for hard reasoning tasks / reasoning-prone models
  * disable thinking for simple chat and creative generation (songs, poems)
  * always reserve most of ``max_tokens`` for the visible answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from seiso.chat.prompts import is_reasoning_prone_model
from seiso.env import env_int, env_str
from seiso.inference.streaming import estimate_chunk_tokens

# Absolute ceilings — never let thinking dominate a pass.
_THINK_MAX_HARD = 256
_THINK_MAX_DEFAULT = 128
_THINK_BUDGET_RATIO_DEFAULT = 0.25
# Visible answer always keeps at least this fraction of the content budget.
_CONTENT_RESERVE_RATIO_DEFAULT = 0.70

_LEVELS = frozenset({"low", "medium", "high", "max"})
_FALSEY = frozenset({"0", "false", "no", "off"})
_TRUEY = frozenset({"1", "true", "yes", "on"})

# Hard multi-step / analytical asks benefit from short internal reasoning.
_COMPLEX_RE = re.compile(
    r"(?is)\b("
    r"prove|derive|theorem|integral|equation|solve\b|debug|traceback|"
    r"implement|refactor|algorithm|complexity|optimize|benchmark|"
    r"step[- ]by[- ]step|reason\b|analyze|analyse|compare|contrast|"
    r"why does|how does|explain (?:why|how)|root cause|trade-?off|"
    r"architecture|design pattern|proof|lemma|formal"
    r")\b"
)
# Creative / short-chat quality is usually better without long monologues.
_CREATIVE_RE = re.compile(
    r"(?is)\b("
    r"song|lyrics|poem|poetry|haiku|story|screenplay|script|rap|"
    r"rhyme|verse|ballad|limerick|joke|pun"
    r")\b"
)
_SIMPLE_RE = re.compile(
    r"(?is)^\s*("
    r"hi|hey|hello|thanks|thank you|ok|okay|yes|no|yep|nope|"
    r"longer|more|again|continue|go|go on|\?|\.\.\."
    r")\s*[.!]?\s*$"
)

_THINK_OPEN_RE = re.compile(r"<(?:redacted_thinking|think)\b[^>]*>", re.I)
_THINK_CLOSE_RE = re.compile(r"</(?:redacted_thinking|think)>", re.I)


@dataclass(frozen=True)
class ThinkingPolicy:
    """Resolved thinking plan for one generation pass."""

    enabled: bool
    # Value for Ollama / API ``think`` field (bool or level string).
    api_value: bool | str
    # Max internal-thinking tokens before abort → content-only recovery.
    think_max_tokens: int
    # Visible-answer budget (OOM-clamped content size).
    content_max_tokens: int
    # Decode budget sent to the backend (content + reserved thinking).
    decode_max_tokens: int
    # auto | on | off | forced
    mode: str
    # none | simple | creative | complex | reasoning_model
    reason: str

    @property
    def thinking_enabled(self) -> bool:
        return self.enabled and self.think_max_tokens > 0


def _latest_user_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
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
            return " ".join(parts).strip()
        return str(content or "").strip()
    return ""


def classify_task(messages: list[dict[str, Any]] | None) -> str:
    """Return a coarse task class for quality-oriented thinking defaults."""
    text = _latest_user_text(messages)
    if not text:
        return "simple"
    if _SIMPLE_RE.match(text) or len(text) < 8:
        return "simple"
    if _CREATIVE_RE.search(text) and not _COMPLEX_RE.search(text):
        return "creative"
    if _COMPLEX_RE.search(text) or len(text) > 400:
        return "complex"
    return "general"


def _parse_mode(raw: str) -> str:
    text = (raw or "").strip().lower()
    if text in _FALSEY or text == "off":
        return "off"
    if text in _TRUEY or text == "on":
        return "on"
    if text in _LEVELS:
        return text  # treat level as on-with-level
    if text in {"auto", ""}:
        return "auto"
    return "auto"


def _parse_api_value(raw: Any) -> bool | str | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in _LEVELS:
        return text
    if text in _FALSEY:
        return False
    if text in _TRUEY:
        return True
    return bool(text)


def thinking_max_tokens(content_max_tokens: int, *, task: str = "general") -> int:
    """Cap internal thinking tokens for a pass (shared by all backends)."""
    content = max(1, int(content_max_tokens or 1))
    # Prefer unified env; fall back to legacy Ollama-specific names.
    raw_cap = env_int(
        "SEISO_THINK_MAX_TOKENS",
        env_int("SEISO_OLLAMA_THINK_MAX_TOKENS", _THINK_MAX_DEFAULT),
    )
    if raw_cap <= 0:
        return 0
    try:
        ratio = float(
            env_str(
                "SEISO_THINK_BUDGET_RATIO",
                env_str(
                    "SEISO_OLLAMA_THINK_BUDGET_RATIO",
                    str(_THINK_BUDGET_RATIO_DEFAULT),
                ),
            )
        )
    except ValueError:
        ratio = _THINK_BUDGET_RATIO_DEFAULT
    ratio = max(0.0, min(0.5, ratio))  # never more than half the content budget
    # Quality scaling: complex gets full ratio; creative/simple get less.
    if task in {"simple", "creative"}:
        ratio *= 0.5
    elif task == "complex":
        ratio = min(0.35, ratio * 1.2)
    by_ratio = int(content * ratio) if ratio > 0 else 0
    if by_ratio <= 0:
        return 0
    try:
        reserve = float(
            env_str(
                "SEISO_CONTENT_RESERVE_RATIO",
                str(_CONTENT_RESERVE_RATIO_DEFAULT),
            )
        )
    except ValueError:
        reserve = _CONTENT_RESERVE_RATIO_DEFAULT
    reserve = max(0.5, min(0.95, reserve))
    # Hard guarantee: content keeps ≥ reserve of the pass.
    max_from_reserve = max(0, int(content * (1.0 - reserve)))
    if max_from_reserve <= 0:
        return 0
    return max(
        1,
        min(int(raw_cap), by_ratio, max_from_reserve, _THINK_MAX_HARD, content),
    )


def resolve_thinking_policy(
    *,
    content_max_tokens: int,
    messages: list[dict[str, Any]] | None = None,
    model_key: str | None = None,
    payload: dict[str, Any] | None = None,
    n_ctx: int | None = None,
) -> ThinkingPolicy:
    """Compute a quality-first, backend-agnostic thinking plan."""
    content = max(1, int(content_max_tokens or 1))
    payload = payload or {}
    task = classify_task(messages)
    model = str(
        model_key
        or payload.get("model_name")
        or payload.get("model_id")
        or payload.get("model_path")
        or ""
    )

    # Explicit request wins.
    if "think" in payload:
        api = _parse_api_value(payload.get("think"))
        if api is None:
            # Omit field: treat as off for budgeting (host default elsewhere).
            return ThinkingPolicy(
                enabled=False,
                api_value=False,
                think_max_tokens=0,
                content_max_tokens=content,
                decode_max_tokens=content,
                mode="forced",
                reason="payload_omit",
            )
        enabled = api is not False and api != 0
        think_max = thinking_max_tokens(content, task=task) if enabled else 0
        decode = _decode_budget(content, think_max, n_ctx=n_ctx)
        return ThinkingPolicy(
            enabled=enabled,
            api_value=api if enabled else False,
            think_max_tokens=think_max,
            content_max_tokens=content,
            decode_max_tokens=decode,
            mode="forced",
            reason="payload",
        )

    # Unified mode env; legacy SEISO_OLLAMA_THINK still honored.
    mode_raw = env_str("SEISO_THINK_MODE", "")
    if not mode_raw:
        mode_raw = env_str("SEISO_OLLAMA_THINK", "auto")
    mode = _parse_mode(mode_raw)

    if mode == "off":
        return ThinkingPolicy(
            enabled=False,
            api_value=False,
            think_max_tokens=0,
            content_max_tokens=content,
            decode_max_tokens=content,
            mode="off",
            reason="env_off",
        )

    level_api: bool | str = True
    if mode in _LEVELS:
        level_api = mode
        mode = "on"

    if mode == "on":
        think_max = thinking_max_tokens(content, task=task if task != "simple" else "general")
        return ThinkingPolicy(
            enabled=think_max > 0,
            api_value=level_api if think_max > 0 else False,
            think_max_tokens=think_max,
            content_max_tokens=content,
            decode_max_tokens=_decode_budget(content, think_max, n_ctx=n_ctx),
            mode="on",
            reason="env_on",
        )

    # --- auto (quality default) ---
    # Creative + simple chat: no thinking (better fluency / no empty burns).
    if task in {"simple", "creative"}:
        return ThinkingPolicy(
            enabled=False,
            api_value=False,
            think_max_tokens=0,
            content_max_tokens=content,
            decode_max_tokens=content,
            mode="auto",
            reason=task,
        )

    reasoning_model = is_reasoning_prone_model(model)
    if task == "complex" or reasoning_model:
        # Prefer a low API level when possible (GPT-OSS / newer Ollama).
        api: bool | str = "low" if reasoning_model else True
        if task == "complex" and reasoning_model:
            api = "medium"
        think_max = thinking_max_tokens(
            content,
            task="complex" if task == "complex" else "general",
        )
        if think_max <= 0:
            return ThinkingPolicy(
                enabled=False,
                api_value=False,
                think_max_tokens=0,
                content_max_tokens=content,
                decode_max_tokens=content,
                mode="auto",
                reason="budget_zero",
            )
        return ThinkingPolicy(
            enabled=True,
            api_value=api,
            think_max_tokens=think_max,
            content_max_tokens=content,
            decode_max_tokens=_decode_budget(content, think_max, n_ctx=n_ctx),
            mode="auto",
            reason="complex" if task == "complex" else "reasoning_model",
        )

    # General chat on non-reasoning models: skip thinking for quality + speed.
    return ThinkingPolicy(
        enabled=False,
        api_value=False,
        think_max_tokens=0,
        content_max_tokens=content,
        decode_max_tokens=content,
        mode="auto",
        reason="general",
    )


def _decode_budget(content: int, think_max: int, *, n_ctx: int | None) -> int:
    if think_max <= 0:
        return content
    predict = content + think_max
    if n_ctx is not None and int(n_ctx) > 0:
        room = max(content, int(n_ctx) - 64)
        predict = min(predict, room)
    return max(content, predict)


def apply_thinking_policy(payload: dict[str, Any]) -> dict[str, Any]:
    """Stamp a resolved thinking plan onto an inference payload (idempotent)."""
    out = dict(payload)
    if out.get("_thinking_policy_applied"):
        return out
    content = max(1, int(out.get("max_tokens") or 512))
    messages = out.get("messages") if isinstance(out.get("messages"), list) else None
    model_key = (
        out.get("model_name")
        or out.get("model_id")
        or out.get("model_path")
        or out.get("model")
    )
    n_ctx = out.get("n_ctx") or out.get("sidecar_num_ctx")
    try:
        n_ctx_i = int(n_ctx) if n_ctx is not None else None
    except (TypeError, ValueError):
        n_ctx_i = None
    policy = resolve_thinking_policy(
        content_max_tokens=content,
        messages=messages,
        model_key=str(model_key) if model_key else None,
        payload=out,
        n_ctx=n_ctx_i,
    )
    # Content budget stays as max_tokens (OOM clamp target); decode expansion
    # is backend-specific (Ollama num_predict) via think_max_tokens stamp.
    out["think"] = policy.api_value
    out["think_max_tokens"] = policy.think_max_tokens
    out["thinking_content_max_tokens"] = policy.content_max_tokens
    out["thinking_decode_max_tokens"] = policy.decode_max_tokens
    out["thinking_mode"] = policy.mode
    out["thinking_reason"] = policy.reason
    out["_thinking_policy_applied"] = True
    return out


def reasoning_quality_system_suffix(*, thinking_enabled: bool) -> str:
    """Short system addendum for better final-answer quality."""
    if thinking_enabled:
        return (
            "Keep any internal reasoning brief and focused. "
            "Always finish with a complete, high-quality visible answer; "
            "do not let planning consume the whole reply."
        )
    return (
        "Answer directly with a complete, high-quality response. "
        "Avoid long hidden planning; put substance in the visible answer."
    )


@dataclass
class ThinkingStreamGuard:
    """Backend-agnostic mid-stream thinking budget enforcer.

    Handles:
      * Ollama-style separate ``thinking`` field chunks
      * Inline ``<think>...</think>`` tags in plain text streams (llama.cpp / HF)
    """

    think_max_tokens: int
    thinking_tokens: int = 0
    saw_visible_content: bool = False
    capped: bool = False
    _in_inline_think: bool = False
    _carry: str = ""

    def feed_thinking_field(self, text: str) -> bool:
        """Account for a dedicated thinking delta. Returns True if budget exhausted."""
        if self.capped or not text or self.saw_visible_content:
            return self.capped
        if self.think_max_tokens <= 0:
            return False
        self.thinking_tokens += estimate_chunk_tokens(text)
        if self.thinking_tokens >= self.think_max_tokens:
            self.capped = True
        return self.capped

    def feed_text(self, text: str) -> tuple[str, bool]:
        """Filter a raw text delta.

        Returns ``(emit_text, abort)``. ``abort`` means thinking exhausted the
        budget before any visible content — caller should stop generation and
        surface a length finish so multi-pass content recovery can run.
        """
        if not text:
            return "", False
        if self.think_max_tokens <= 0:
            # No cap: still strip nothing here (sanitizer handles tags for UI).
            if text.strip():
                self.saw_visible_content = True
            return text, False

        buf = self._carry + text
        self._carry = ""
        emit_parts: list[str] = []
        i = 0
        while i < len(buf):
            if self._in_inline_think:
                close = _THINK_CLOSE_RE.search(buf, i)
                if close is None:
                    # Rest is still thinking; count and maybe abort.
                    rest = buf[i:]
                    # Hold back a short suffix that might be a partial close tag.
                    hold = min(len(rest), 24)
                    body, maybe_partial = rest[:-hold] if hold else rest, rest[-hold:] if hold else ""
                    if body:
                        self.thinking_tokens += estimate_chunk_tokens(body)
                    self._carry = maybe_partial
                    if self.thinking_tokens >= self.think_max_tokens and not self.saw_visible_content:
                        self.capped = True
                        return "".join(emit_parts), True
                    return "".join(emit_parts), False
                # Thinking segment ends.
                think_piece = buf[i : close.start()]
                if think_piece:
                    self.thinking_tokens += estimate_chunk_tokens(think_piece)
                i = close.end()
                self._in_inline_think = False
                if self.thinking_tokens >= self.think_max_tokens and not self.saw_visible_content:
                    self.capped = True
                    return "".join(emit_parts), True
                continue

            open_m = _THINK_OPEN_RE.search(buf, i)
            if open_m is None:
                # Possible partial open tag at end.
                tail = buf[i:]
                # Keep a short holdback for partial ``<think``.
                hold_n = 0
                for n in range(1, min(len(tail), 20) + 1):
                    frag = tail[-n:].lower()
                    if "<think".startswith(frag) or "<redacted_thinking".startswith(frag) or frag.endswith("<"):
                        hold_n = n
                        break
                visible = tail[:-hold_n] if hold_n else tail
                self._carry = tail[-hold_n:] if hold_n else ""
                if visible:
                    emit_parts.append(visible)
                    if visible.strip():
                        self.saw_visible_content = True
                break

            if open_m.start() > i:
                visible = buf[i : open_m.start()]
                if visible:
                    emit_parts.append(visible)
                    if visible.strip():
                        self.saw_visible_content = True
            i = open_m.end()
            self._in_inline_think = True

        return "".join(emit_parts), False

    def stats(self) -> dict[str, Any]:
        return {
            "thinking_tokens": self.thinking_tokens,
            "think_max_tokens": self.think_max_tokens,
            "thinking_capped": self.capped,
            "saw_visible_content": self.saw_visible_content,
        }


# Back-compat aliases used by older Ollama helpers / tests.
def ollama_think_max_tokens(content_max_tokens: int) -> int:
    return thinking_max_tokens(content_max_tokens, task="general")
