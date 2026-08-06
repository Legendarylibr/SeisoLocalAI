"""Speculative decoding — draft proposes, target verifies (Leviathan et al. 2022)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from seiso.env import env_bool, env_int
from seiso.inference.streaming import StreamToken

logger = logging.getLogger(__name__)

_DEFAULT_SPECULATIVE_TOKENS = 5


@dataclass(frozen=True)
class TorchSpeculativeBundle:
    target_model: Any
    target_tokenizer: Any
    draft_model: Any
    draft_tokenizer: Any


@dataclass(frozen=True)
class DFlashDraftSpeculativeBundle:
    """Target (usually torch) + dflash GGUF draft loaded via llama.cpp for fast proposals."""

    target_model: Any
    target_tokenizer: Any
    draft_llm: Any  # llama_cpp.Llama instance for the dflash draft
    draft_tokenizer: Any  # usually same as target or compatible


def default_num_speculative_tokens(payload: dict[str, Any]) -> int:
    raw = payload.get("num_speculative_tokens")
    if raw is not None:
        return max(1, int(raw))
    return max(1, env_int("SEISO_SPECULATIVE_TOKENS", _DEFAULT_SPECULATIVE_TOKENS))


def _use_speculative_kv_cache() -> bool:
    return env_bool("SEISO_SPECULATIVE_KV_CACHE", True)


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _decode_new_text(tokenizer: Any, token_ids: Any, prev_char_len: int) -> tuple[str, int]:
    """Decode the full sequence and return only the new text suffix (BPE-safe)."""
    text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    return text[prev_char_len:], len(text)


def _model_forward(model: Any, input_ids: Any, *, past_key_values: Any = None) -> Any:
    kwargs: dict[str, Any] = {"use_cache": True}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    return model(input_ids, **kwargs)


def _kv_cache_usable(out: Any) -> bool:
    return getattr(out, "past_key_values", None) is not None


_KV_CACHE_FALLBACK_ERRORS = (TypeError, AttributeError, RuntimeError, ValueError)


def _crop_past_key_values(past_key_values: Any, seq_len: int) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(seq_len)
        return past_key_values
    return past_key_values


def _can_crop_past_key_values(past_key_values: Any) -> bool:
    return past_key_values is not None and hasattr(past_key_values, "crop")


def _verify_proposed(
    prefix_logits: Any,
    verify_logits: Any,
    proposed_ids_t: Any,
) -> int:
    import torch

    k = int(proposed_ids_t.shape[1])
    if k == 0:
        return 0
    preds = proposed_ids_t.new_empty((1, k))
    preds[:, 0] = prefix_logits.argmax(dim=-1)
    if k > 1:
        preds[:, 1:] = verify_logits[:, : k - 1, :].argmax(dim=-1)
    match = (preds == proposed_ids_t).to(torch.int32)
    return int(match.cumprod(dim=1).sum(dim=1).item())


def _target_ids_for_continuation(
    target_tokenizer: Any, current_text: str, continuation: str
) -> list[int]:
    """Map draft continuation text into target token ids without BPE junction splits.

    Encoding ``continuation`` alone can diverge at the boundary vs encoding the
    full ``current_text + continuation`` string; take the suffix delta instead.
    """
    if not continuation:
        return []
    encode = getattr(target_tokenizer, "encode", None)
    if encode is None:
        ids = target_tokenizer(continuation, add_special_tokens=False)
        raw = ids["input_ids"] if isinstance(ids, dict) else ids
        return list(raw)[: len(continuation)]
    full = list(encode(current_text + continuation, add_special_tokens=False))
    prefix = list(encode(current_text, add_special_tokens=False))
    if len(full) >= len(prefix) and full[: len(prefix)] == prefix:
        return full[len(prefix) :]
    # Fallback when prefix is not an exact token prefix (rare tokenizer quirks).
    return list(encode(continuation, add_special_tokens=False))


def _propose_with_dflash_draft(
    draft_llm: Any,
    target_tokenizer: Any,
    current_text: str,
    k: int,
    temperature: float = 0.0,
) -> list[int]:
    """Use fast dflash (llama.cpp) draft to propose k tokens, return token ids in *target* tokenizer space."""
    from seiso.inference.model_pool import dflash_draft_infer

    proposed_text = dflash_draft_infer(
        draft_llm,
        current_text,
        max_tokens=k,
        temperature=temperature,
    )
    if not proposed_text:
        return []

    proposed_ids = _target_ids_for_continuation(target_tokenizer, current_text, proposed_text)
    return proposed_ids[:k]


def _iter_speculative_tokens_naive(
    *,
    bundle: TorchSpeculativeBundle,
    prompt: str,
    max_new_tokens: int,
    num_speculative_tokens: int,
    temperature: float = 0.0,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[StreamToken]:
    import torch

    target = bundle.target_model
    draft = bundle.draft_model
    tok = bundle.target_tokenizer
    draft_tok = bundle.draft_tokenizer or tok
    shared_vocab = draft_tok is tok

    target_device = _model_device(target)
    draft_device = _model_device(draft)

    input_ids_t = tok(prompt, return_tensors="pt").input_ids.to(target_device)
    input_ids_d = (
        input_ids_t.to(draft_device)
        if shared_vocab
        else draft_tok(prompt, return_tensors="pt").input_ids.to(draft_device)
    )

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

            if shared_vocab:
                proposed_ids_t = (
                    torch.cat(proposed, dim=1).to(target_device) if proposed else input_ids_t[:, :0]
                )
            else:
                # Bridge via text + prefix-delta encode into target vocab.
                draft_piece = (
                    draft_tok.decode(torch.cat(proposed, dim=1)[0], skip_special_tokens=True)
                    if proposed
                    else ""
                )
                current_text = tok.decode(input_ids_t[0], skip_special_tokens=True)
                mapped = _target_ids_for_continuation(tok, current_text, draft_piece)[:k]
                proposed_ids_t = (
                    torch.tensor([mapped], device=target_device, dtype=input_ids_t.dtype)
                    if mapped
                    else input_ids_t[:, :0]
                )
                k = int(proposed_ids_t.shape[1])
            candidate = torch.cat([input_ids_t, proposed_ids_t], dim=1)

            t_out = target(candidate)
            logits = t_out.logits

            prefix_len = input_ids_t.shape[1]
            verify_slice = logits[:, prefix_len : prefix_len + max(k, 1) - 1, :]
            accept = _verify_proposed(
                logits[:, prefix_len - 1, :],
                verify_slice,
                proposed_ids_t,
            )

            if accept > 0:
                input_ids_t = torch.cat([input_ids_t, proposed_ids_t[:, :accept]], dim=1)
                if shared_vocab:
                    input_ids_d = input_ids_t.to(draft_device)
                else:
                    input_ids_d = draft_tok(
                        tok.decode(input_ids_t[0], skip_special_tokens=True),
                        return_tensors="pt",
                    ).input_ids.to(draft_device)
                tokens_generated += accept
                chunk, decoded_len = _decode_new_text(tok, input_ids_t, decoded_len)
                if chunk:
                    yield StreamToken(chunk, accept)

            if tokens_generated >= max_new_tokens or stop():
                break

            pos = prefix_len + accept - 1 if accept < k else prefix_len + k - 1
            next_id = torch.argmax(logits[:, pos, :], dim=-1, keepdim=True).to(target_device)
            input_ids_t = torch.cat([input_ids_t, next_id], dim=1)
            if shared_vocab:
                input_ids_d = input_ids_t.to(draft_device)
            else:
                input_ids_d = draft_tok(
                    tok.decode(input_ids_t[0], skip_special_tokens=True),
                    return_tensors="pt",
                ).input_ids.to(draft_device)
            tokens_generated += 1
            chunk, decoded_len = _decode_new_text(tok, input_ids_t, decoded_len)
            if chunk:
                yield StreamToken(chunk)


def _iter_speculative_tokens_cached(
    *,
    bundle: TorchSpeculativeBundle,
    prompt: str,
    max_new_tokens: int,
    num_speculative_tokens: int,
    temperature: float = 0.0,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[StreamToken]:
    import torch

    target = bundle.target_model
    draft = bundle.draft_model
    tok = bundle.target_tokenizer
    draft_tok = bundle.draft_tokenizer or tok
    # Cached path requires shared token ids; fall back when draft vocab differs.
    if draft_tok is not tok:
        yield from _iter_speculative_tokens_naive(
            bundle=bundle,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_speculative_tokens=num_speculative_tokens,
            temperature=temperature,
            should_stop=should_stop,
        )
        return

    target_device = _model_device(target)
    draft_device = _model_device(draft)

    input_ids_t = tok(prompt, return_tensors="pt").input_ids.to(target_device)
    input_ids_d = input_ids_t.to(draft_device)

    tokens_generated = 0
    decoded_len = len(tok.decode(input_ids_t[0], skip_special_tokens=True))
    stop = should_stop or (lambda: False)

    with torch.inference_mode():
        t_out = _model_forward(target, input_ids_t)
        if not _kv_cache_usable(t_out):
            raise RuntimeError("target model returned no past_key_values")
        target_past = t_out.past_key_values
        prefix_logits = t_out.logits[:, -1, :]

        d_out = _model_forward(draft, input_ids_d)
        if not _kv_cache_usable(d_out):
            raise RuntimeError("draft model returned no past_key_values")
        draft_past = d_out.past_key_values
        draft_prefix_logits = d_out.logits[:, -1, :]

        while tokens_generated < max_new_tokens:
            if stop():
                break

            k = min(num_speculative_tokens, max_new_tokens - tokens_generated)
            saved_prefix_logits = prefix_logits

            proposed: list[Any] = []
            next_logits = draft_prefix_logits
            for _ in range(k):
                if temperature > 0:
                    next_logits = next_logits / max(temperature, 0.01)
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                proposed.append(next_id)
                input_ids_d = torch.cat([input_ids_d, next_id.to(draft_device)], dim=1)
                d_out = _model_forward(draft, next_id.to(draft_device), past_key_values=draft_past)
                if not _kv_cache_usable(d_out):
                    raise RuntimeError("draft model returned no past_key_values")
                draft_past = d_out.past_key_values
                next_logits = d_out.logits[:, -1, :]

            proposed_ids_t = torch.cat(proposed, dim=1).to(target_device)

            v_out = _model_forward(target, proposed_ids_t, past_key_values=target_past)
            verify_logits = v_out.logits
            accept = _verify_proposed(saved_prefix_logits, verify_logits, proposed_ids_t)

            if accept == k:
                input_ids_t = torch.cat([input_ids_t, proposed_ids_t], dim=1)
                target_past = v_out.past_key_values
                prefix_logits = verify_logits[:, k - 1, :]
                draft_prefix_logits = next_logits
                tokens_generated += k
                chunk, decoded_len = _decode_new_text(tok, input_ids_t, decoded_len)
                if chunk:
                    yield StreamToken(chunk, k)
                continue

            if accept > 0:
                seq_len = int(input_ids_t.shape[1]) + accept
                if hasattr(v_out.past_key_values, "crop"):
                    target_past = _crop_past_key_values(v_out.past_key_values, seq_len)
                else:
                    a_out = _model_forward(
                        target,
                        proposed_ids_t[:, :accept],
                        past_key_values=target_past,
                    )
                    target_past = a_out.past_key_values
                input_ids_t = torch.cat([input_ids_t, proposed_ids_t[:, :accept]], dim=1)
                tokens_generated += accept
                chunk, decoded_len = _decode_new_text(tok, input_ids_t, decoded_len)
                if chunk:
                    yield StreamToken(chunk, accept)

            if tokens_generated >= max_new_tokens or stop():
                break

            next_logits = saved_prefix_logits if accept == 0 else verify_logits[:, accept - 1, :]

            next_id = torch.argmax(next_logits, dim=-1, keepdim=True).to(target_device)
            input_ids_t = torch.cat([input_ids_t, next_id], dim=1)

            c_out = _model_forward(target, next_id, past_key_values=target_past)
            target_past = c_out.past_key_values
            prefix_logits = c_out.logits[:, -1, :]

            input_ids_d = input_ids_t.to(draft_device)
            if _can_crop_past_key_values(draft_past):
                draft_prefix_len = int(input_ids_t.shape[1]) - 1
                draft_past = _crop_past_key_values(draft_past, draft_prefix_len)
                d_out = _model_forward(
                    draft,
                    next_id.to(draft_device),
                    past_key_values=draft_past,
                )
            else:
                d_out = _model_forward(draft, input_ids_d)
            if not _kv_cache_usable(d_out):
                raise RuntimeError("draft model returned no past_key_values")
            draft_past = d_out.past_key_values
            draft_prefix_logits = d_out.logits[:, -1, :]

            tokens_generated += 1
            chunk, decoded_len = _decode_new_text(tok, input_ids_t, decoded_len)
            if chunk:
                yield StreamToken(chunk)


def iter_speculative_tokens(
    *,
    bundle: TorchSpeculativeBundle,
    prompt: str,
    max_new_tokens: int,
    num_speculative_tokens: int,
    temperature: float = 0.0,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[StreamToken]:
    """
    Stream decoded text from speculative decoding.

    Greedy verification when temperature <= 0; draft uses temperature scaling when > 0.
    """
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")

    if _use_speculative_kv_cache():
        emitted = False
        try:
            for token in _iter_speculative_tokens_cached(
                bundle=bundle,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_speculative_tokens=num_speculative_tokens,
                temperature=temperature,
                should_stop=should_stop,
            ):
                emitted = True
                yield token
            return
        except _KV_CACHE_FALLBACK_ERRORS as exc:
            if emitted:
                raise RuntimeError(
                    "Speculative KV cache failed after streaming began — "
                    "aborting instead of replaying partial output"
                ) from exc
            logger.debug("Speculative KV cache unavailable — falling back: %s", exc)

    yield from _iter_speculative_tokens_naive(
        bundle=bundle,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_speculative_tokens=num_speculative_tokens,
        temperature=temperature,
        should_stop=should_stop,
    )


def _iter_speculative_tokens_dflash_cached(
    *,
    bundle: DFlashDraftSpeculativeBundle,
    prompt: str,
    max_new_tokens: int,
    num_speculative_tokens: int,
    temperature: float = 0.0,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[StreamToken]:
    import torch

    target = bundle.target_model
    target_tok = bundle.target_tokenizer
    draft_llm = bundle.draft_llm

    target_device = _model_device(target)

    input_ids_t = target_tok(prompt, return_tensors="pt").input_ids.to(target_device)

    tokens_generated = 0
    current_text = target_tok.decode(input_ids_t[0], skip_special_tokens=True)
    decoded_len = len(current_text)
    stop = should_stop or (lambda: False)

    with torch.inference_mode():
        t_out = _model_forward(target, input_ids_t)
        if not _kv_cache_usable(t_out):
            raise RuntimeError("target model returned no past_key_values")
        target_past = t_out.past_key_values
        prefix_logits = t_out.logits[:, -1, :]

        while tokens_generated < max_new_tokens:
            if stop():
                break

            k = min(num_speculative_tokens, max_new_tokens - tokens_generated)
            saved_prefix_logits = prefix_logits

            proposed_ids = _propose_with_dflash_draft(
                draft_llm,
                target_tok,
                current_text,
                k,
                temperature=temperature,
            )

            if not proposed_ids:
                next_id = torch.argmax(prefix_logits, dim=-1, keepdim=True).to(target_device)
                input_ids_t = torch.cat([input_ids_t, next_id], dim=1)
                c_out = _model_forward(target, next_id, past_key_values=target_past)
                target_past = c_out.past_key_values
                prefix_logits = c_out.logits[:, -1, :]
                tokens_generated += 1
                chunk, decoded_len = _decode_new_text(target_tok, input_ids_t, decoded_len)
                current_text += chunk
                if chunk:
                    yield StreamToken(chunk)
                continue

            proposed_ids_t = torch.tensor([proposed_ids], device=target_device)

            v_out = _model_forward(target, proposed_ids_t, past_key_values=target_past)
            verify_logits = v_out.logits
            accept = _verify_proposed(saved_prefix_logits, verify_logits, proposed_ids_t)

            if accept == len(proposed_ids):
                input_ids_t = torch.cat([input_ids_t, proposed_ids_t], dim=1)
                target_past = v_out.past_key_values
                prefix_logits = verify_logits[:, accept - 1, :]
                tokens_generated += accept
                chunk, decoded_len = _decode_new_text(target_tok, input_ids_t, decoded_len)
                current_text += chunk
                if chunk:
                    yield StreamToken(chunk, accept)
                continue

            if accept > 0:
                seq_len = int(input_ids_t.shape[1]) + accept
                if hasattr(v_out.past_key_values, "crop"):
                    target_past = _crop_past_key_values(v_out.past_key_values, seq_len)
                else:
                    a_out = _model_forward(
                        target,
                        proposed_ids_t[:, :accept],
                        past_key_values=target_past,
                    )
                    target_past = a_out.past_key_values
                input_ids_t = torch.cat([input_ids_t, proposed_ids_t[:, :accept]], dim=1)
                tokens_generated += accept
                chunk, decoded_len = _decode_new_text(target_tok, input_ids_t, decoded_len)
                current_text += chunk
                if chunk:
                    yield StreamToken(chunk, accept)

            if tokens_generated >= max_new_tokens or stop():
                break

            next_logits = saved_prefix_logits if accept == 0 else verify_logits[:, accept - 1, :]

            next_id = torch.argmax(next_logits, dim=-1, keepdim=True).to(target_device)
            input_ids_t = torch.cat([input_ids_t, next_id], dim=1)

            c_out = _model_forward(target, next_id, past_key_values=target_past)
            target_past = c_out.past_key_values
            prefix_logits = c_out.logits[:, -1, :]

            tokens_generated += 1
            chunk, decoded_len = _decode_new_text(target_tok, input_ids_t, decoded_len)
            current_text += chunk
            if chunk:
                yield StreamToken(chunk)


def _iter_speculative_tokens_dflash_naive(
    *,
    bundle: DFlashDraftSpeculativeBundle,
    prompt: str,
    max_new_tokens: int,
    num_speculative_tokens: int,
    temperature: float = 0.0,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[StreamToken]:
    import torch

    target = bundle.target_model
    target_tok = bundle.target_tokenizer
    draft_llm = bundle.draft_llm

    target_device = _model_device(target)

    input_ids_t = target_tok(prompt, return_tensors="pt").input_ids.to(target_device)

    tokens_generated = 0
    current_text = target_tok.decode(input_ids_t[0], skip_special_tokens=True)
    decoded_len = len(current_text)
    stop = should_stop or (lambda: False)

    with torch.inference_mode():
        while tokens_generated < max_new_tokens:
            if stop():
                break

            k = min(num_speculative_tokens, max_new_tokens - tokens_generated)

            proposed_ids = _propose_with_dflash_draft(
                draft_llm,
                target_tok,
                current_text,
                k,
                temperature=temperature,
            )

            if not proposed_ids:
                t_out = target(input_ids_t)
                next_id = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True).to(
                    target_device
                )
                input_ids_t = torch.cat([input_ids_t, next_id], dim=1)
                tokens_generated += 1
                chunk, decoded_len = _decode_new_text(target_tok, input_ids_t, decoded_len)
                current_text += chunk
                if chunk:
                    yield StreamToken(chunk)
                continue

            proposed_ids_t = torch.tensor([proposed_ids], device=target_device)
            candidate = torch.cat([input_ids_t, proposed_ids_t], dim=1)

            t_out = target(candidate)
            logits = t_out.logits

            prefix_len = input_ids_t.shape[1]
            k = len(proposed_ids)
            verify_slice = logits[:, prefix_len : prefix_len + k - 1, :]
            accept = _verify_proposed(
                logits[:, prefix_len - 1, :],
                verify_slice,
                proposed_ids_t,
            )

            if accept > 0:
                input_ids_t = torch.cat([input_ids_t, proposed_ids_t[:, :accept]], dim=1)
                tokens_generated += accept
                chunk, decoded_len = _decode_new_text(target_tok, input_ids_t, decoded_len)
                current_text += chunk
                if chunk:
                    yield StreamToken(chunk, accept)

            if tokens_generated >= max_new_tokens or stop():
                break

            pos = (
                prefix_len + accept - 1
                if accept < len(proposed_ids)
                else prefix_len + len(proposed_ids) - 1
            )
            next_id = torch.argmax(logits[:, pos, :], dim=-1, keepdim=True).to(target_device)
            input_ids_t = torch.cat([input_ids_t, next_id], dim=1)
            tokens_generated += 1
            chunk, decoded_len = _decode_new_text(target_tok, input_ids_t, decoded_len)
            current_text += chunk
            if chunk:
                yield StreamToken(chunk)


def iter_speculative_tokens_dflash(
    *,
    bundle: DFlashDraftSpeculativeBundle,
    prompt: str,
    max_new_tokens: int,
    num_speculative_tokens: int,
    temperature: float = 0.0,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[StreamToken]:
    """Speculative decoding using dflash GGUF draft (via llama.cpp for speed) + target verifier."""
    if num_speculative_tokens < 1:
        raise ValueError("num_speculative_tokens must be >= 1")

    if _use_speculative_kv_cache():
        emitted = False
        try:
            for token in _iter_speculative_tokens_dflash_cached(
                bundle=bundle,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_speculative_tokens=num_speculative_tokens,
                temperature=temperature,
                should_stop=should_stop,
            ):
                emitted = True
                yield token
            return
        except _KV_CACHE_FALLBACK_ERRORS as exc:
            if emitted:
                raise RuntimeError(
                    "dFlash speculative KV cache failed after streaming began — "
                    "aborting instead of replaying partial output"
                ) from exc
            logger.debug("dFlash speculative KV cache unavailable — falling back: %s", exc)

    yield from _iter_speculative_tokens_dflash_naive(
        bundle=bundle,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_speculative_tokens=num_speculative_tokens,
        temperature=temperature,
        should_stop=should_stop,
    )
