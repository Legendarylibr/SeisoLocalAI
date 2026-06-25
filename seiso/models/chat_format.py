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
        if "query" in sample and "response" in sample:
            return [
                {"role": "user", "content": str(sample.get("query") or "")},
                {"role": "assistant", "content": str(sample.get("response") or "")},
            ]
        if "question" in sample and "answer" in sample:
            return [
                {"role": "user", "content": str(sample.get("question") or "")},
                {"role": "assistant", "content": str(sample.get("answer") or "")},
            ]
        if "prompt" in sample and ("completion" in sample or "response" in sample):
            return [
                {"role": "user", "content": str(sample.get("prompt") or "")},
                {
                    "role": "assistant",
                    "content": str(sample.get("completion") or sample.get("response") or ""),
                },
            ]
        instruction = sample.get("instruction", "")
        inp = sample.get("input", "")
        output = sample.get("output") or sample.get("response") or ""
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

    if fmt == DatasetFormat.PREFERENCE:
        from seiso.training.preprocess import parse_human_assistant_dialog

        if sample.get("messages"):
            return cast(list[dict[str, Any]], sample["messages"])
        chosen = sample.get("chosen") or sample.get("chosen_response") or sample.get("accepted")
        messages = parse_human_assistant_dialog(chosen)
        if messages:
            return messages
        prompt = str(sample.get("prompt") or "")
        response = str(chosen or "")
        if prompt and response:
            return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        return []

    return [{"role": "user", "content": sample.get("text") or sample.get("content") or str(sample)}]
