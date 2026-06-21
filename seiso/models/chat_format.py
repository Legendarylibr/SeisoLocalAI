"""Shared chat formatting for training and inference."""

from __future__ import annotations

from typing import Any, cast


def format_messages_for_prompt(
    messages: list[dict],
    tokenizer,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render chat messages to a single prompt string."""
    if hasattr(tokenizer, "apply_chat_template"):
        return cast(
            str,
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            ),
        )
    parts = [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages]
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n".join(parts)


def extract_messages(sample: dict[str, Any], fmt) -> list[dict[str, Any]]:
    """Extract normalized message list from a training sample."""
    from seiso.training.config import DatasetFormat

    if fmt == DatasetFormat.ALPACA:
        instruction = sample.get("instruction", "")
        inp = sample.get("input", "")
        output = sample.get("output", "")
        user = f"{instruction}\n{inp}".strip() if inp else instruction
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]

    if fmt == DatasetFormat.SHAREGPT and "conversations" in sample:
        messages = []
        for turn in sample["conversations"]:
            role = turn.get("from", turn.get("role", "user"))
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "bot"):
                role = "assistant"
            messages.append({"role": role, "content": turn.get("value", turn.get("content", ""))})
        return messages

    if fmt == DatasetFormat.CHAT:
        return cast(list[dict[str, Any]], sample.get("messages", []))

    return [{"role": "user", "content": sample.get("text") or sample.get("content") or str(sample)}]
