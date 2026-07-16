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
# Prompt builders often end with an open ``<think>``; the model then continues
# the body and emits only ``</think>`` plus the final answer.
_THINK_CONTINUATION_RE = re.compile(
    r"^(?P<trace>.*?)</think>(?P<final>.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
_CHOICE_RE = re.compile(r"\b([a-d])\b", flags=re.IGNORECASE)
# Explicit final-answer letter patterns (preferred over first free-form letter).
_CHOICE_ANSWER_MARKERS = (
    re.compile(
        r"(?:final\s+answer|answer|choice|option|select(?:ed)?)\s*(?:is|:|=)?\s*"
        r"[\(\[]?\s*([a-d])\s*[\)\].:]?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*[\(\[]?\s*([a-d])\s*[\)\]\.\:]\s*(?:\n|$)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*([a-d])\s*$",
        flags=re.IGNORECASE,
    ),
)
# Prefer numbers after explicit answer markers / boxed / final line.
_NUMBER_ANSWER_MARKERS = (
    re.compile(
        r"(?:final\s+answer|answer|result|equals?)\s*(?:is|:|=)?\s*"
        r"([-+]?(?:\d*\.\d+|\d+))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\\boxed\{\s*([-+]?(?:\d*\.\d+|\d+))\s*\}",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*([-+]?(?:\d*\.\d+|\d+))\s*$",
    ),
)
# Trailing confidence / count phrases often pollute last-number extraction.
_NOISE_NUMBER_CONTEXT_RE = re.compile(
    r"(?:confidence|certainty|percent|%|cases?|steps?|times?)\s*"
    r"[-+]?(?:\d*\.\d+|\d+)"
    r"|"
    r"[-+]?(?:\d*\.\d+|\d+)\s*(?:%|percent|cases?|steps?|times?)",
    flags=re.IGNORECASE,
)


def format_thinking_prompt(prompt: str, instruction: str) -> str:
    """Append a thinking instruction and opening tag unless already present."""
    if "<think>" in prompt.lower():
        return prompt
    return f"{prompt.rstrip()}\n\n{instruction}\n<think>"


def split_thinking_trace(completion: str) -> tuple[str, str, bool]:
    """Split completion into (thinking_trace, final_answer, has_closed_trace).

    Closed format is accepted when either:
    - the completion contains a full ``<think>...</think>`` block, or
    - the completion continues a prompt-opened think and contains ``</think>``
      (trace before the close, final answer after).

    An open ``<think>`` without a close still yields ``has_closed_trace=False``.
    """
    match = _THINK_SPLIT_RE.search(completion)
    if match is not None:
        return match.group("trace").strip(), match.group("final").strip(), True

    # Continuation of a prompt that already emitted ``<think>``.
    cont = _THINK_CONTINUATION_RE.match(completion)
    if cont is not None and _THINK_CLOSE_RE.search(completion):
        # Reject pure open-tag-only false positives: require an actual close tag.
        return cont.group("trace").strip(), cont.group("final").strip(), True

    open_match = _THINK_OPEN_RE.search(completion)
    if open_match is not None:
        return open_match.group("trace").strip(), "", False
    return "", completion.strip(), False


def has_closed_thinking_trace(completion: str) -> bool:
    """True when thinking is closed in the *generated* text.

    Accepts either a full ``<think>...</think>`` block or a prompt-continuation
    close (``</think>`` after reasoning, without re-opening the tag).
    """
    _, _, closed = split_thinking_trace(completion)
    return closed


def final_answer_text(completion: str) -> str:
    """Text after the last ``</think>``, else tag-stripped completion."""
    match = re.search(
        r"</think>(?P<final>.*)$",
        completion,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        return match.group("final").strip()
    return _THINK_BLOCK_RE.sub("", completion).strip()


def last_number(text: str) -> float | None:
    """Extract the most likely final numeric answer from free-form text.

    Preference order:
    1. Numbers after explicit answer markers / ``\\boxed{}`` / final-line only.
    2. Last number in the text after stripping confidence/count noise phrases.
    3. Last raw number as a last resort.
    """
    cleaned = text.replace(",", "")
    for pattern in _NUMBER_ANSWER_MARKERS:
        matches = pattern.findall(cleaned)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    denoised = _NOISE_NUMBER_CONTEXT_RE.sub(" ", cleaned)
    matches = _NUMBER_RE.findall(denoised)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass

    matches = _NUMBER_RE.findall(cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def extract_choice(text: str) -> str | None:
    """Extract a multiple-choice letter ``a``–``d`` from free-form text.

    Prefers explicit answer markers and the last letter in the final-answer
    span over the first free-form ``\\b[a-d]\\b`` match (avoids “A is wrong… B”).
    """
    if not text or not str(text).strip():
        return None
    source = str(text)

    # Collect all marker hits and take the rightmost (final) answer cue.
    marker_hits: list[tuple[int, str]] = []
    for pattern in _CHOICE_ANSWER_MARKERS:
        for match in pattern.finditer(source):
            marker_hits.append((match.start(), match.group(1).lower()))
    if marker_hits:
        marker_hits.sort(key=lambda item: item[0])
        return marker_hits[-1][1]

    # Unique letter in the whole span — unambiguous.
    all_letters = [m.group(1).lower() for m in _CHOICE_RE.finditer(source)]
    if not all_letters:
        return None
    unique = list(dict.fromkeys(all_letters))
    if len(unique) == 1:
        return unique[0]

    # Multiple free-form letters: take the last one (usually the final pick).
    return all_letters[-1]


def normalize_answer(text: str) -> str:
    lowered = final_answer_text(text).lower()
    lowered = re.sub(r"final answer\s*(is|:)?", " ", lowered)
    lowered = re.sub(r"answer\s*(is|:)?", " ", lowered)
    lowered = re.sub(r"[^a-z0-9.+-]+", " ", lowered)
    return " ".join(lowered.split()).strip()
