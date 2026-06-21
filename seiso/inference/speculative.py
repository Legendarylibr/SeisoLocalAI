"""Speculative decoding — draft proposes, target verifies (Leviathan et al. 2022)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from seiso.env import env_int

logger = logging.getLogger(__name__)

_DEFAULT_SPECULATIVE_TOKENS = 5


@dataclass(frozen=True)
class TorchSpeculativeBundle:
    target_model: Any
    target_tokenizer: Any
    draft_model: Any
    draft_tokenizer: Any


def default_num_speculative_tokens(payload: dict[str, Any]) -> int:
    raw = payload.get("num_speculative_tokens")
    if raw is not None:
        return max(1, int(raw))
    return max(1, env_int("SEISO_SPECULATIVE_TOKENS", _DEFAULT_SPECULATIVE_TOKENS))


def _model_device(model: Any) -> Any:

    return next(model.parameters()).device


def _decode_new_text(tokenizer: Any, token_ids: Any, prev_len: int) -> tuple[str, int]:
    text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    return text[prev_len:], len(text)


def iter_speculative_tokens(
    *,
    bundle: TorchSpeculativeBundle,
    prompt: str,
    max_new_tokens: int,
    num_speculative_tokens: int,
    temperature: float = 0.0,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[str]:
    """
    Stream decoded text from speculative decoding.

    Greedy verification when temperature <= 0; draft uses temperature scaling when > 0.
    """
    import torch

    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")

    target = bundle.target_model
    draft = bundle.draft_model
    tok = bundle.target_tokenizer

    target_device = _model_device(target)
    draft_device = _model_device(draft)

    input_ids_t = tok(prompt, return_tensors="pt").input_ids.to(target_device)
    input_ids_d = input_ids_t.to(draft_device)

    tokens_generated = 0
    decoded_len = len(tok.decode(input_ids_t[0], skip_special_tokens=True))
    stop = should_stop or (lambda: False)

    with torch.inference_mode():
        while tokens_generated < max_new_tokens:
            if stop():
                break

            k = min(num_speculative_tokens, max_new_tokens - tokens_generated)

            draft_ids = input_ids_d
            proposed: list[Any] = []
            for _ in range(k):
                d_out = draft(draft_ids)
                next_logits = d_out.logits[:, -1, :]
                if temperature > 0:
                    next_logits = next_logits / max(temperature, 0.01)
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                proposed.append(next_id)
                draft_ids = torch.cat([draft_ids, next_id.to(draft_ids.device)], dim=1)

            proposed_ids_d = torch.cat(proposed, dim=1) if proposed else input_ids_d[:, :0]
            proposed_ids_t = proposed_ids_d.to(target_device)
            candidate = torch.cat([input_ids_t, proposed_ids_t], dim=1)

            t_out = target(candidate)
            logits = t_out.logits

            prefix_len = input_ids_t.shape[1]
            accept = 0
            for j in range(k):
                pos = prefix_len + j - 1
                greedy = torch.argmax(logits[:, pos, :], dim=-1)
                if int(greedy.item()) == int(proposed_ids_t[0, j].item()):
                    accept += 1
                else:
                    break

            if accept > 0:
                input_ids_t = torch.cat([input_ids_t, proposed_ids_t[:, :accept]], dim=1)
                input_ids_d = input_ids_t.to(draft_device)
                tokens_generated += accept
                chunk, decoded_len = _decode_new_text(tok, input_ids_t, decoded_len)
                if chunk:
                    yield chunk

            if tokens_generated >= max_new_tokens or stop():
                break

            pos = prefix_len + accept - 1 if accept < k else prefix_len + k - 1
            next_id = torch.argmax(logits[:, pos, :], dim=-1, keepdim=True).to(target_device)
            input_ids_t = torch.cat([input_ids_t, next_id], dim=1)
            input_ids_d = input_ids_t.to(draft_device)
            tokens_generated += 1
            chunk, decoded_len = _decode_new_text(tok, input_ids_t, decoded_len)
            if chunk:
                yield chunk
