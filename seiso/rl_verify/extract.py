"""Text extraction helpers for verifiable RL scoring."""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_THINK_SPLIT_RE = re.compile(
    r"<think>(?P<trace>.*?)</think>(?P<final>.*)",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_OPEN_RE = re.compile(
    r"<think>(?P<trace>.*)",
    flags=re.IGNORECASE | re.DOTALL,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
_CHOICE_RE = re.compile(r"\b([a-d])\b", flags=re.IGNORECASE)


def format_thinking_prompt(prompt: str, instruction: str) -> str:
    """Append a thinking instruction and opening tag unless already present."""
    if "<think>" in prompt.lower():
        return prompt
    return f"{prompt.rstrip()}\n\n{instruction}\n<think>"


def split_thinking_trace(completion: str) -> tuple[str, str, bool]:
    """Split completion into (thinking_trace, final_answer, has_closed_trace).

    Only a closed ``</think>`` counts as a complete format. Open tags return
    ``has_closed_trace=False`` so format rewards never invent structure.
    """
    match = _THINK_SPLIT_RE.search(completion)
    if match is not None:
        return match.group("trace").strip(), match.group("final").strip(), True
    open_match = _THINK_OPEN_RE.search(completion)
    if open_match is not None:
        return open_match.group("trace").strip(), "", False
    return "", completion.strip(), False


def has_closed_thinking_trace(completion: str) -> bool:
    """True when the generated text contains a closed ``<think>...</think>`` block."""
    return _THINK_SPLIT_RE.search(completion) is not None


def final_answer_text(completion: str) -> str:
    """Text after the last closed thinking block, else tag-stripped completion."""
    match = re.search(
        r"</think>(?P<final>.*)$",
        completion,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        return match.group("final").strip()
    return _THINK_BLOCK_RE.sub("", completion).strip()


def last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def extract_choice(text: str) -> str | None:
    match = _CHOICE_RE.search(text)
    return match.group(1).lower() if match else None


def normalize_answer(text: str) -> str:
    lowered = final_answer_text(text).lower()
    lowered = re.sub(r"final answer\s*(is|:)?", " ", lowered)
    lowered = re.sub(r"answer\s*(is|:)?", " ", lowered)
    lowered = re.sub(r"[^a-z0-9.+-]+", " ", lowered)
    return " ".join(lowered.split()).strip()
