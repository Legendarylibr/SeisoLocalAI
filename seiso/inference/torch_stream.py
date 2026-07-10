"""Manual torch decode loop with past_key_values (cooperative cancel).

Falls back to TextIteratorStreamer + generate when the model does not expose
usable KV caches or sampling needs HF generate features.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from seiso.env import env_bool, env_int
from seiso.inference.streaming import StreamToken
from seiso.memory.protection import is_oom_error, release_cached_memory

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PrefixState:
    key: str
    token_ids: list[int]
    past_key_values: Any
    logits: Any


_PREFIX_LOCK = threading.RLock()
_PREFIX_STATE: _PrefixState | None = None


def clear_torch_prefix_cache(cache_key: str | None = None) -> None:
    """Drop the bounded process-wide Torch prefix state."""
    global _PREFIX_STATE  # pylint: disable=global-statement
    with _PREFIX_LOCK:
        if cache_key is None or (_PREFIX_STATE is not None and _PREFIX_STATE.key == cache_key):
            _PREFIX_STATE = None


def _prefix_hit(cache_key: str, token_ids: list[int]) -> _PrefixState | None:
    with _PREFIX_LOCK:
        state = _PREFIX_STATE
        if (
            state is not None
            and state.key == cache_key
            and len(state.token_ids) <= len(token_ids)
            and token_ids[: len(state.token_ids)] == state.token_ids
        ):
            return state
    return None


def _store_prefix(cache_key: str, token_ids: list[int], past_key_values: Any, logits: Any) -> None:
    global _PREFIX_STATE  # pylint: disable=global-statement
    if len(token_ids) > max(256, env_int("SEISO_TORCH_PREFIX_CACHE_MAX_TOKENS", 32768)):
        clear_torch_prefix_cache(cache_key)
        return
    with _PREFIX_LOCK:
        _PREFIX_STATE = _PrefixState(
            key=cache_key,
            token_ids=list(token_ids),
            past_key_values=past_key_values,
            logits=logits,
        )


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
    prefill_chunk_size: int | None = None,
    cache_key: str | None = None,
    prefix_cache: bool = False,
    stats: dict[str, Any] | None = None,
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
    logits = None
    do_sample = temperature is not None and float(temperature) > 0
    completed = False
    metrics = stats if stats is not None else {}
    metrics.update(
        {
            "prefill_chunks": 0,
            "prefix_hit": False,
            "prefill_backoffs": 0,
        }
    )

    try:
        with torch.inference_mode():
            prefill_started = time.perf_counter()
            state = (
                _prefix_hit(cache_key, token_ids)
                if prefix_cache and cache_key is not None
                else None
            )
            if state is not None:
                prefix_len = len(state.token_ids)
                past = state.past_key_values
                logits = state.logits
                metrics["prefix_hit"] = True
            else:
                prefix_len = 0

            remaining = input_ids[:, prefix_len:]
            configured = max(
                1,
                int(prefill_chunk_size or remaining.shape[-1] or input_ids.shape[-1]),
            )
            chunk_size = min(configured, max(1, int(remaining.shape[-1])))
            while remaining.shape[-1] > 0:
                try:
                    offset = 0
                    while offset < remaining.shape[-1]:
                        chunk = remaining[:, offset : offset + chunk_size]
                        out = _model_forward(model, chunk, past_key_values=past)
                        if not _kv_cache_usable(out):
                            raise RuntimeError("model returned no past_key_values")
                        past = out.past_key_values
                        logits = out.logits[:, -1, :]
                        metrics["prefill_chunks"] += 1
                        offset += int(chunk.shape[-1])
                    break
                except Exception as exc:
                    if not is_oom_error(exc) or chunk_size <= 1:
                        clear_torch_prefix_cache(cache_key)
                        if is_oom_error(exc):
                            raise RuntimeError("chunk prefill exhausted adaptive retries") from exc
                        raise
                    metrics["prefill_backoffs"] += 1
                    chunk_size = max(1, chunk_size // 2)
                    past = None
                    logits = None
                    prefix_len = 0
                    remaining = input_ids
                    release_cached_memory(sync=True)

            if logits is None or past is None:
                raise RuntimeError("model returned no usable KV prefill state")
            metrics["prefill_ms"] = round((time.perf_counter() - prefill_started) * 1000.0, 3)
            metrics["prefill_chunk_size"] = chunk_size

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
                    completed = True
                    break

                token_ids.append(token_int)
                step = _model_forward(model, next_id, past_key_values=past)
                past = step.past_key_values
                logits = step.logits[:, -1, :]
                tokens_generated += 1
                if pad_token_id is not None and token_int == int(pad_token_id):
                    continue
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                chunk, decoded_len = text[decoded_len:], len(text)
                if chunk:
                    yield StreamToken(chunk)
            else:
                completed = True
    except Exception:
        clear_torch_prefix_cache(cache_key)
        raise
    finally:
        if stop():
            clear_torch_prefix_cache(cache_key)
        elif completed and prefix_cache and cache_key is not None and past is not None:
            _store_prefix(cache_key, token_ids, past, logits)
