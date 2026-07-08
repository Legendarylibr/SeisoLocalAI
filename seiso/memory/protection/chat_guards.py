"""Chat payload trimming and sanitization."""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any

from seiso.env import env_bool, env_int
from seiso.memory.protection._facade import protection
from seiso.memory.protection.constants import (
    _INFERENCE_OVERHEAD_MB,
    _MAX_INFERENCE_TOKENS,
    _NATIVE_LINUX_LOW_HEADROOM_MAX_COMPLETION_TOKENS,
    _NATIVE_LINUX_MAX_COMPLETION_TOKENS,
    _NATIVE_LINUX_PREFILL_CLAMP_MB,
)

_VISION_TOKENS_PER_IMAGE = 1024
_DATA_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,", re.I)
_VISION_CONTENT_MARKERS = (
    "image_url",
    '"type":"image"',
    '"type": "image"',
    "data:image/",
)
_CONTEXT_TRIM_MARKER = "[...older content omitted...]\n"


def _text_chars_to_tokens(chars: int) -> int:
    return max(0, int(chars / 3.2))


def _count_images_in_content(content: Any) -> int:
    if isinstance(content, list):
        count = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "")).lower()
            if part_type in {"image", "image_url"}:
                count += 1
        return count
    if not isinstance(content, str):
        return 0
    stripped = content.lstrip()
    if stripped.startswith("["):
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return _count_images_in_content(parsed)
    lower = content.lower()
    embedded = len(_DATA_IMAGE_RE.findall(content))
    if embedded:
        return embedded
    if any(marker in lower for marker in _VISION_CONTENT_MARKERS):
        return max(1, lower.count("image_url"))
    return 0


def _text_chars_from_content(content: Any) -> int:
    if isinstance(content, list):
        chars = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "text")).lower()
            if part_type in {"text", "input_text"}:
                chars += len(str(part.get("text") or part.get("content") or ""))
        return chars
    if isinstance(content, str):
        stripped = content.lstrip()
        if stripped.startswith("["):
            with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return _text_chars_from_content(parsed)
        if _count_images_in_content(content):
            # OpenAI-style JSON with embedded base64 — avoid treating payload as text.
            with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return _text_chars_from_content(parsed)
            return min(len(content), 512)
        return len(content)
    return len(str(content))


def _message_content_token_estimate(content: Any) -> int:
    images = _count_images_in_content(content)
    text_tokens = _text_chars_to_tokens(_text_chars_from_content(content))
    if images:
        return text_tokens + images * _VISION_TOKENS_PER_IMAGE
    return text_tokens


def _messages_have_vision_content(messages: list[dict[str, Any]]) -> bool:
    return any(_count_images_in_content(m.get("content")) > 0 for m in messages)


def _gguf_has_mmproj_sibling(model_path: str | Path) -> bool:
    """True when a colocated mmproj GGUF is present for a vision-capable chat model."""
    path = Path(model_path)
    if not path.is_file():
        return False
    from seiso.inference.llama_vision import model_suggests_vision, resolve_mmproj_path

    if not model_suggests_vision(path):
        return False
    return resolve_mmproj_path(path) is not None


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    total = sum(_message_content_token_estimate(m.get("content", "")) for m in messages)
    return max(64, total)


def _trim_text_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    char_budget = max(1, int(token_budget * 3.2))
    if len(text) <= char_budget:
        return text
    if char_budget <= len(_CONTEXT_TRIM_MARKER):
        return text[-char_budget:]
    keep = char_budget - len(_CONTEXT_TRIM_MARKER)
    return f"{_CONTEXT_TRIM_MARKER}{text[-keep:]}"


def _trim_message_content_to_token_budget(content: Any, token_budget: int) -> Any:
    if isinstance(content, str):
        return _trim_text_to_token_budget(content, token_budget)
    if isinstance(content, list):
        remaining = max(0, token_budget)
        out: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                out.append(part)
                continue
            part_type = str(part.get("type", "text")).lower()
            if part_type in {"image", "image_url"}:
                if remaining >= _VISION_TOKENS_PER_IMAGE:
                    out.append(part)
                    remaining = max(0, remaining - _VISION_TOKENS_PER_IMAGE)
                continue
            if part_type not in {"text", "input_text"}:
                out.append(part)
                continue
            text = str(part.get("text") or part.get("content") or "")
            trimmed = _trim_text_to_token_budget(text, remaining)
            remaining = max(0, remaining - _text_chars_to_tokens(len(trimmed)))
            key = "text" if "text" in part else "content"
            out.append({**part, key: trimmed})
        return out
    return content


def trim_llama_messages_to_context(
    messages: list[dict[str, Any]],
    *,
    n_ctx: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Trim prompt content so llama.cpp prefill stays within the loaded context."""
    if not messages:
        return []

    prompt_budget = max(256, int(n_ctx) - max(1, int(max_tokens)) - 128)
    if _estimate_prompt_tokens(messages) <= prompt_budget:
        return messages

    trimmed = [dict(message) for message in messages]
    latest_idx = len(trimmed) - 1

    # Drop oldest conversational turns first; keep system/knowledge instructions
    # until content trimming is required.
    idx = 0
    while _estimate_prompt_tokens(trimmed) > prompt_budget and idx < latest_idx:
        role = str(trimmed[idx].get("role", "")).lower()
        if role in {"user", "assistant"}:
            trimmed.pop(idx)
            latest_idx -= 1
            continue
        idx += 1

    # Then trim oversized message bodies, newest user last.
    order = sorted(
        range(len(trimmed)),
        key=lambda i: (
            i == len(trimmed) - 1,
            str(trimmed[i].get("role", "")).lower() == "system" and i == 0,
            -_message_content_token_estimate(trimmed[i].get("content", "")),
        ),
    )
    for idx in order:
        current = _estimate_prompt_tokens(trimmed)
        if current <= prompt_budget:
            break
        content = trimmed[idx].get("content", "")
        content_tokens = _message_content_token_estimate(content)
        if content_tokens <= 0:
            continue
        target = max(32, content_tokens - (current - prompt_budget))
        trimmed[idx]["content"] = _trim_message_content_to_token_budget(content, target)

    return trimmed


def sanitize_inference_payload(
    payload: dict[str, Any], *, isolated: bool = False
) -> dict[str, Any]:
    """Clamp generation limits to available memory without changing intent.

    ``isolated=True`` means the request is served by an out-of-process sidecar
    (Ollama/llama-swap) that manages its own memory and cannot crash Forge, so
    the in-process VRAM-motivated completion caps are skipped and only the
    absolute token ceiling applies.
    """
    out = dict(payload)
    messages = out.get("messages") or []
    prompt_tokens = _estimate_prompt_tokens(messages)
    headroom = protection().headroom_mb()

    max_tokens = int(out.get("max_tokens") or 2048)
    max_tokens = max(1, min(max_tokens, _MAX_INFERENCE_TOKENS))

    if not isolated and headroom > _INFERENCE_OVERHEAD_MB:
        kv_budget_tokens = max(
            512,
            int((headroom - _INFERENCE_OVERHEAD_MB) * 128 / 1.15),
        )
        max_tokens = min(max_tokens, max(128, kv_budget_tokens - prompt_tokens - 32))
    try:
        from seiso.platform import use_linux_nvidia_inference_guards

        native_linux_nvidia = use_linux_nvidia_inference_guards()
    except Exception:
        native_linux_nvidia = False
    if (
        native_linux_nvidia
        and not isolated
        and not env_bool("SEISO_LLAMA_UNSAFE_LONG_COMPLETIONS", False)
    ):
        native_cap = env_int(
            "SEISO_LLAMA_NATIVE_MAX_TOKENS",
            _NATIVE_LINUX_MAX_COMPLETION_TOKENS,
        )
        if headroom < _NATIVE_LINUX_PREFILL_CLAMP_MB:
            native_cap = min(native_cap, _NATIVE_LINUX_LOW_HEADROOM_MAX_COMPLETION_TOKENS)
        max_tokens = min(max_tokens, max(1, native_cap))
    out["max_tokens"] = max_tokens

    if out.get("n_ctx") is not None:
        from seiso.memory.protection.llama_clamp import clamp_llama_n_ctx

        out["n_ctx"] = clamp_llama_n_ctx(
            int(out["n_ctx"]),
            messages=messages,
            max_tokens=max_tokens,
            model_path=out.get("model_path"),
            model_format=out.get("model_format"),
        )
    return out


