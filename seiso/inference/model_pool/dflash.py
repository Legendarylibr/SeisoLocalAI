"""DFlash / draft model cache for speculative decoding."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from seiso.inference.model_pool._facade import model_pool as _mp

logger = logging.getLogger(__name__)


class DflashDraftHandle:
    """Thread-safe wrapper around a cached llama.cpp dflash/draft model."""

    __slots__ = ("llm", "n_ctx", "_infer_lock", "_last_prompt", "_last_tokens")

    def __init__(self, llm: Any, n_ctx: int = 0) -> None:
        self.llm = llm
        self.n_ctx = n_ctx
        self._infer_lock = threading.Lock()
        self._last_prompt = ""
        self._last_tokens: list[int] = []

    def dispose(self) -> None:
        """Close the native handle only after in-flight infer finishes."""
        with self._infer_lock:
            llm = self.llm
            self.llm = None
            self._last_prompt = ""
            self._last_tokens = []
            if llm is None:
                return
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                logger.debug("Failed to close dflash draft handle", exc_info=True)


_dflash_draft_cache: dict[str, DflashDraftHandle] = {}
_dflash_draft_lock = threading.Lock()
_dflash_key_locks: dict[str, threading.Lock] = {}
_dflash_key_locks_guard = threading.Lock()


def _load_dflash_llm(resolved_path: str, n_ctx: int) -> Any:
    return _mp()._load_llama_model(resolved_path, n_ctx)


def _dflash_lock_for(norm: str) -> threading.Lock:
    with _dflash_key_locks_guard:
        return _dflash_key_locks.setdefault(norm, threading.Lock())


def get_dflash_draft(model_path: str, *, n_ctx: int = 4096) -> DflashDraftHandle:
    """Return a cached, thread-safe llama.cpp handle for dflash/draft GGUF models."""
    from seiso.inference.backends import BACKEND_LLAMACPP, prepare_model_path

    resolved = prepare_model_path(model_path, BACKEND_LLAMACPP)
    norm = str(Path(resolved).resolve())
    with _dflash_lock_for(norm):
        with _dflash_draft_lock:
            cached = _dflash_draft_cache.get(norm)
            if cached is not None and cached.n_ctx >= n_ctx and cached.llm is not None:
                return cached

        llm = _mp()._load_dflash_llm(resolved, n_ctx)

        old_cached: DflashDraftHandle | None = None
        with _dflash_draft_lock:
            cached = _dflash_draft_cache.get(norm)
            if cached is not None and cached.n_ctx >= n_ctx and cached.llm is not None:
                try:
                    if hasattr(llm, "close"):
                        llm.close()
                except Exception:
                    logger.debug("Failed to close duplicate dflash draft", exc_info=True)
                return cached
            if cached is not None:
                # Drop from cache first so new callers do not receive a disposed handle.
                _dflash_draft_cache.pop(norm, None)
                old_cached = cached
            handle = DflashDraftHandle(llm, n_ctx=n_ctx)
            _dflash_draft_cache[norm] = handle
        if old_cached is not None:
            old_cached.dispose()
        return handle


def _dflash_completion(
    llm: Any,
    current_text: str,
    gen_kwargs: dict[str, Any],
    *,
    reuse_prefix: bool,
) -> dict[str, Any]:
    """Call llama.cpp completion, preferring prompt-cache reuse when extending a prefix."""
    if reuse_prefix:
        try:
            return llm(current_text, cache_prompt=True, **gen_kwargs)
        except TypeError:
            # Older llama-cpp-python builds may not accept cache_prompt.
            pass
        except Exception:
            logger.debug("dflash cache_prompt failed; retrying cold", exc_info=True)
    return llm(current_text, **gen_kwargs)


def _dflash_tokenize(llm: Any, text: str) -> list[int] | None:
    """Best-effort tokenize via llama.cpp; None when unavailable."""
    tokenize = getattr(llm, "tokenize", None)
    if not callable(tokenize):
        return None
    try:
        tokens = tokenize(text.encode("utf-8"), add_bos=False)
    except TypeError:
        try:
            tokens = tokenize(text.encode("utf-8"))
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(tokens, (list, tuple)):
        return None
    return [int(t) for t in tokens]


def _dflash_completion_tokens(
    llm: Any,
    tokens: list[int],
    gen_kwargs: dict[str, Any],
    *,
    reuse_prefix: bool,
) -> dict[str, Any] | None:
    """Token-ID completion path (avoids re-encoding growing text each round)."""
    try:
        if reuse_prefix:
            try:
                return llm(tokens, cache_prompt=True, **gen_kwargs)
            except TypeError:
                pass
            except Exception:
                logger.debug("dflash token cache_prompt failed; retrying cold", exc_info=True)
        return llm(tokens, **gen_kwargs)
    except TypeError:
        return None
    except Exception:
        logger.debug("dflash token completion failed; falling back to text", exc_info=True)
        return None


def dflash_draft_infer(
    draft: Any,
    current_text: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Run a single dflash draft completion under the per-handle inference lock.

    When ``draft`` is a :class:`DflashDraftHandle` and ``current_text`` extends the
    previous prompt, llama.cpp ``cache_prompt`` is requested so the draft KV can
    be reused across speculative rounds. Prefer a token-ID path when the draft
    tokenizer is available so growing text is not re-encoded every round.
    """
    if isinstance(draft, DflashDraftHandle):
        llm = draft.llm
        infer_lock = draft._infer_lock
        handle: DflashDraftHandle | None = draft
    else:
        llm = draft
        infer_lock = None
        handle = None

    gen_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "echo": False,
        "temperature": max(temperature, 0.0) if temperature and temperature > 0 else 0.0,
    }
    if not temperature or temperature <= 0:
        gen_kwargs["temperature"] = 0.0

    def _run(active_llm: Any) -> str:
        reuse_text = bool(
            handle is not None
            and handle._last_prompt
            and current_text.startswith(handle._last_prompt)
        )
        tokens: list[int] | None = None
        if handle is not None:
            if (
                reuse_text
                and handle._last_tokens
                and current_text.startswith(handle._last_prompt)
            ):
                suffix = current_text[len(handle._last_prompt) :]
                if not suffix:
                    tokens = list(handle._last_tokens)
                else:
                    suffix_tokens = _dflash_tokenize(active_llm, suffix)
                    if suffix_tokens is not None:
                        tokens = list(handle._last_tokens) + suffix_tokens
            if tokens is None:
                tokens = _dflash_tokenize(active_llm, current_text)

        if tokens is not None:
            reuse_tokens = bool(
                handle is not None
                and handle._last_tokens
                and len(tokens) >= len(handle._last_tokens)
                and tokens[: len(handle._last_tokens)] == handle._last_tokens
            )
            out = _dflash_completion_tokens(
                active_llm, tokens, gen_kwargs, reuse_prefix=reuse_tokens
            )
            if out is not None:
                if handle is not None:
                    handle._last_prompt = current_text
                    handle._last_tokens = tokens
                return out["choices"][0]["text"] if out.get("choices") else ""

        out = _dflash_completion(
            active_llm, current_text, gen_kwargs, reuse_prefix=reuse_text
        )
        if handle is not None:
            handle._last_prompt = current_text
            if tokens is not None:
                handle._last_tokens = tokens
        return out["choices"][0]["text"] if out.get("choices") else ""

    if infer_lock is not None:
        with infer_lock:
            llm = draft.llm
            if llm is None:
                return ""
            return _run(llm)
    return _run(llm)


def clear_dflash_draft_cache() -> None:
    """Release cached dflash draft models."""
    with _dflash_draft_lock:
        handles = list(_dflash_draft_cache.values())
        _dflash_draft_cache.clear()
    for handle in handles:
        handle.dispose()
