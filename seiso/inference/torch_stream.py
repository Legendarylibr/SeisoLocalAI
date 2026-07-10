"""Manual torch decode loop with past_key_values (cooperative cancel).

Falls back to TextIteratorStreamer + generate when the model does not expose
usable KV caches or sampling needs HF generate features.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from seiso.env import env_bool
from seiso.inference.streaming import StreamToken

logger = logging.getLogger(__name__)


def use_manual_torch_kv_stream(payload: dict[str, Any] | None = None) -> bool:
    """Whether to prefer the cooperative KV decode loop (default on)."""
    if payload and payload.get("torch_kv_stream") is not None:
        return bool(payload["torch_kv_stream"])
    return env_bool("SEISO_TORCH_KV_STREAM", True)


def iter_torch_kv_tokens(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    pad_token_id: int | None = None,
    eos_token_id: int | list[int] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[StreamToken]:
    """Greedy / simple sampling decode with incremental past_key_values."""
    import torch

    from seiso.inference.speculative import (
        _kv_cache_usable,
        _model_forward,
    )

    stop = should_stop or (lambda: False)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_token_id}
    else:
        eos_ids = set()

    tokens_generated = 0
    # Keep the display sequence on CPU. Repeatedly torch.cat-ing the full
    # sequence on GPU copies an ever-growing tensor every decode step and can
    # fragment VRAM; the model itself only needs next_id after prefill.
    token_ids = input_ids[0].detach().cpu().tolist()
    decoded_len = len(tokenizer.decode(token_ids, skip_special_tokens=True))
    past = None
    do_sample = temperature is not None and float(temperature) > 0

    with torch.inference_mode():
        out = _model_forward(model, input_ids)
        if not _kv_cache_usable(out):
            raise RuntimeError("model returned no past_key_values")
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        while tokens_generated < max_new_tokens:
            if stop():
                break
            if do_sample:
                scaled = logits / max(float(temperature), 0.01)
                if top_p is not None and 0 < float(top_p) < 1:
                    sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    mask = cum > float(top_p)
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    probs = torch.softmax(sorted_logits, dim=-1)
                    choice = torch.multinomial(probs, num_samples=1)
                    next_id = sorted_idx.gather(-1, choice)
                else:
                    probs = torch.softmax(scaled, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            token_int = int(next_id.item())
            if token_int in eos_ids:
                break

            token_ids.append(token_int)
            step = _model_forward(model, next_id, past_key_values=past)
            past = step.past_key_values
            logits = step.logits[:, -1, :]
            tokens_generated += 1
            if pad_token_id is not None and token_int == int(pad_token_id):
                # Advance the cache but do not expose padding as output.
                continue
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            chunk, decoded_len = text[decoded_len:], len(text)
            if chunk:
                yield StreamToken(chunk)
