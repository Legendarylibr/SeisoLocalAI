"""DFlash / draft model cache for speculative decoding."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from seiso.inference.model_pool._facade import model_pool as _mp


class DflashDraftHandle:
    """Thread-safe wrapper around a cached llama.cpp dflash/draft model."""

    __slots__ = ("llm", "n_ctx", "_infer_lock")

    def __init__(self, llm: Any, n_ctx: int = 0) -> None:
        self.llm = llm
        self.n_ctx = n_ctx
        self._infer_lock = threading.Lock()

    def dispose(self) -> None:
        """Close the native handle only after in-flight infer finishes."""
        with self._infer_lock:
            llm = self.llm
            self.llm = None
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


def dflash_draft_infer(
    draft: Any,
    current_text: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Run a single dflash draft completion under the per-handle inference lock."""
    if isinstance(draft, DflashDraftHandle):
        llm = draft.llm
        infer_lock = draft._infer_lock
    else:
        llm = draft
        infer_lock = None

    gen_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "echo": False,
        "temperature": max(temperature, 0.0) if temperature and temperature > 0 else 0.0,
    }
    if not temperature or temperature <= 0:
        gen_kwargs["temperature"] = 0.0

    if infer_lock is not None:
        with infer_lock:
            llm = draft.llm
            if llm is None:
                return ""
            out = llm(current_text, **gen_kwargs)
    else:
        out = llm(current_text, **gen_kwargs)

    return out["choices"][0]["text"] if out.get("choices") else ""


def clear_dflash_draft_cache() -> None:
    """Release cached dflash draft models."""
    with _dflash_draft_lock:
        handles = list(_dflash_draft_cache.values())
        _dflash_draft_cache.clear()
    for handle in handles:
        handle.dispose()
