"""VRAM-managed singleton model pool."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from seiso.compat import StrEnum
from seiso.env import env_int, env_str
from seiso.inference.model_pool._facade import model_pool as _mp
from seiso.inference.model_pool.dflash import clear_dflash_draft_cache
from seiso.memory.gpu_resource_lock import (
    acquire_gpu_resource_lock,
    release_gpu_resource_lock,
)

logger = logging.getLogger(__name__)

_POOL_BACKEND_BY_API: dict[str, str] = {
    "llamacpp": "llamacpp",
    "llama": "llamacpp",
    "llamaswap": "llamaswap",
    "mlx": "mlx",
    "torch": "torch",
}


class BackendKind(StrEnum):
    LLAMACPP = "llamacpp"
    LLAMASWAP = "llamaswap"
    MLX = "mlx"
    TORCH = "torch"

    # Backward-compatible alias for older call sites.
    LLAMA = LLAMACPP


_switch_load_lock = threading.Lock()


@dataclass
class LoadedModel:
    key: str
    backend: BackendKind
    handle: Any
    meta: dict = field(default_factory=dict)


class ModelPool:
    """
    Singleton pool holding at most one active inference model.
    Switching models unloads the previous one from VRAM/RAM.
    """

    _instance: ModelPool | None = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._active: LoadedModel | None = None
        self._generation = 0
        self._inference_refs = 0
        self._unload_pending = False
        self._llama_infer_lock = threading.RLock()
        self._idle_cond = threading.Condition(self._lock)
        self._release_notes: list[str] = []
        self._resident_gpu_resource_lock = False

    @classmethod
    def get(cls) -> ModelPool:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls, *, timeout_s: float = 30.0) -> None:
        """Unload and forget the process-wide pool (shutdown/tests)."""
        with cls._lock:
            instance = cls._instance
        if instance is None:
            return
        instance.cancel_and_unload()
        idle = instance._wait_for_inference_idle(timeout_s=timeout_s)
        if not idle:
            logger.warning(
                "ModelPool.reset_instance: inference still active after %.1fs — "
                "refusing to orphan the busy pool",
                timeout_s,
            )
            return
        instance.unload_all()
        with cls._lock:
            if cls._instance is instance:
                cls._instance = None

    @property
    def active_key(self) -> str | None:
        with self._lock:
            return self._active.key if self._active else None

    @property
    def active_inference_refs(self) -> int:
        with self._lock:
            return self._inference_refs

    def has_active_inference(self) -> bool:
        return self.active_inference_refs > 0

    def drain_release_notes(self) -> list[str]:
        with self._lock:
            notes = list(self._release_notes)
            self._release_notes.clear()
        return notes

    @staticmethod
    def normalize_path(model_path: str) -> str:
        return str(Path(model_path).expanduser().resolve())

    def bump_generation(self) -> int:
        """Invalidate in-flight streams (e.g. before loading another model)."""
        with self._lock:
            self._generation += 1
            return self._generation

    def is_generation_active(self, generation_id: int) -> bool:
        with self._lock:
            return generation_id == self._generation

    def begin_inference(self) -> None:
        acquire_gpu_resource_lock()
        with self._lock:
            self._inference_refs += 1

    @contextmanager
    def inference_lease(self) -> Iterator[None]:
        """Hold the pool busy until a load/generation operation fully exits."""
        self.begin_inference()
        try:
            yield
        finally:
            self.end_inference()

    def end_inference(self) -> None:
        should_unload = False
        had_ref = False
        with self._idle_cond:
            had_ref = self._inference_refs > 0
            self._inference_refs = max(0, self._inference_refs - 1)
            if self._inference_refs == 0:
                self._idle_cond.notify_all()
                if self._unload_pending:
                    should_unload = True
        if should_unload:
            self.unload_all()
        if had_ref:
            release_gpu_resource_lock()

    def _ensure_resident_gpu_resource_lock(self) -> None:
        with self._lock:
            if self._resident_gpu_resource_lock:
                return
        acquire_gpu_resource_lock()
        with self._lock:
            if self._resident_gpu_resource_lock:
                release_gpu_resource_lock()
                return
            self._resident_gpu_resource_lock = True

    def _release_resident_gpu_resource_lock(self) -> None:
        with self._lock:
            if not self._resident_gpu_resource_lock:
                return
            self._resident_gpu_resource_lock = False
        release_gpu_resource_lock()

    def acquire_llama_inference(self) -> None:
        """Serialize use of the shared llama.cpp context."""
        self._llama_infer_lock.acquire()

    def release_llama_inference(self) -> None:
        self._llama_infer_lock.release()

    @contextmanager
    def llama_inference_lease(self) -> Iterator[None]:
        """Serialize access to a shared llama.cpp context."""
        self.acquire_llama_inference()
        try:
            yield
        finally:
            self.release_llama_inference()

    def cancel_and_unload(self) -> None:
        """Stop lagging streams and release VRAM/RAM."""
        from seiso.inference.torch_stream import clear_torch_prefix_cache

        self.bump_generation()
        clear_torch_prefix_cache()
        with self._lock:
            self._unload_pending = True
            if self._inference_refs > 0:
                return
        self.unload_all()
        clear_dflash_draft_cache()

    def _wait_for_inference_idle(self, timeout_s: float = 30.0) -> bool:
        """Wait for active completions/streams to release their pool refs.

        Returns True when idle. On timeout, does not force-clear refs.
        """
        deadline = time.time() + timeout_s
        with self._idle_cond:
            while self._inference_refs > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._idle_cond.wait(timeout=remaining)
            idle = self._inference_refs == 0
        if not idle:
            logger.warning(
                "Inference still active after %.1fs — proceeding with forced unload",
                timeout_s,
            )
        return idle

    def _release_handle(self, active: LoadedModel) -> None:
        """Close one pool handle and free GPU caches."""
        from seiso.inference.torch_stream import clear_torch_prefix_cache

        backend = active.backend
        key = active.key
        handle = active.handle
        # Drop the LoadedModel reference before cache clear so GC can reclaim.
        active.handle = None
        logger.info("Unloading model from VRAM: %s", key)

        if backend == BackendKind.LLAMA:
            llm = handle
            handle = None
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                logger.debug("Failed to close llama handle for %s", key, exc_info=True)
            del llm

        elif backend == BackendKind.LLAMASWAP:
            model_path = active.meta.get("path") or active.meta.get("norm_path")
            release = getattr(handle, "release_external_memory", None)
            if callable(release):
                ok, reason = release(str(model_path) if model_path else None)
                if ok:
                    note = "Released llama-swap managed model processes"
                    logger.info(note)
                else:
                    note = "Could not confirm llama-swap external model unload" + (
                        f": {reason}" if reason else ""
                    )
                    logger.warning(note)
                with self._lock:
                    self._release_notes.append(note)
            del handle

        elif backend == BackendKind.TORCH:
            try:
                from seiso.inference.speculative import TorchSpeculativeBundle
                from seiso.kernels.lifecycle import release_training_memory

                if isinstance(handle, TorchSpeculativeBundle):
                    release_training_memory(handle.target_model, sync=False)
                    release_training_memory(handle.draft_model, sync=False)
                else:
                    model = handle[0] if isinstance(handle, tuple) and handle else handle
                    release_training_memory(model, sync=False)
            except Exception:
                logger.exception("Failed to release torch handle for %s", key)
                try:
                    from seiso.kernels.lifecycle import restore_kernel_patches

                    restore_kernel_patches()
                except Exception:
                    logger.exception(
                        "Global kernel restore also failed after torch unload error"
                    )
            del handle

        elif backend == BackendKind.MLX:
            del handle

        clear_torch_prefix_cache()
        self._free_memory(sync=True)
        clear_dflash_draft_cache()
        self._release_resident_gpu_resource_lock()

    def _unload_active_immediate(self) -> None:
        """Drop the active handle without canceling the in-flight request.

        Used by OOM/prefill recovery: the caller already holds an inference ref
        and will replace the handle, so we must not wait on our own ref or
        invalidate generation_id.

        Preserves ``_unload_pending`` so a concurrent ``cancel_and_unload`` that
        only armed the flag (refs > 0) is not forgotten across the reload.
        """
        with self._lock:
            active = self._active
            self._active = None
        if active is not None:
            self._release_handle(active)
        else:
            clear_dflash_draft_cache()

    def _is_same_model_reload(
        self,
        key: str,
        backend: BackendKind,
        norm_path: str | None = None,
    ) -> bool:
        """True when switch() is reloading the same llama.cpp pool entry."""
        if backend != BackendKind.LLAMA:
            return False
        with self._lock:
            active = self._active
        if not active or active.backend != backend:
            return False
        if active.key == key:
            return True
        return bool(norm_path and active.meta.get("norm_path") == norm_path)

    def _clear_active_for_switch(self) -> None:
        """Stop streams, wait for idle, then unload so a new model can load."""
        self.bump_generation()
        idle = self._wait_for_inference_idle()
        with self._lock:
            prior_pending = self._unload_pending
            if not idle and self._inference_refs > 0:
                self._unload_pending = True
                raise RuntimeError(
                    "Inference is still active; retry after the current generation stops"
                )
            # Preserve cancel_and_unload intent across the reload; switch()'s
            # post-load path discards a fresh handle when unload remains pending.
            self._unload_pending = prior_pending
            active = self._active
            self._active = None
        if active is not None:
            self._release_handle(active)
        else:
            clear_dflash_draft_cache()

    def would_switch_model(
        self, target_path: str, backend: str | BackendKind | None = None
    ) -> bool:
        """True when loading target_path would replace the active inference model."""
        with self._lock:
            active = self._active
        if not active:
            return False
        if backend is not None:
            raw = backend.value if isinstance(backend, BackendKind) else str(backend).lower()
            expected = _POOL_BACKEND_BY_API.get(raw, raw)
            if active.backend.value != expected:
                return True
        active_path = active.meta.get("path") or active.meta.get("norm_path")
        if not active_path:
            return False
        norm_target = self.normalize_path(target_path)
        norm_active = self.normalize_path(str(active_path))
        if norm_active != norm_target:
            return True
        # Speculative bundles (spec:target:draft) are not interchangeable with
        # single-model pool handles that share the same target path.
        return active.key.startswith("spec:")

    def prepare_for_load(
        self,
        target_path: str | None = None,
        backend: str | BackendKind | None = None,
    ) -> bool:
        """Unload the active model when switching and refresh GPU memory stats."""
        should_unload = target_path is None or self.would_switch_model(target_path, backend)
        unloaded = False
        if should_unload and self.active_key:
            # Wait for in-flight inference so VRAM is actually freed before load.
            self._clear_active_for_switch()
            unloaded = True
        if unloaded:
            _mp()._clear_optimal_layers_cache()
            _mp()._refresh_headroom_stats(force=True)
        return unloaded

    def _switch_cache_hit(
        self,
        key: str,
        backend: BackendKind,
        load_path: str,
        meta: dict[str, Any],
    ) -> Any | None:
        with self._lock:
            if not self._active or self._active.key != key:
                return None
            needed_ctx = int(meta.get("n_ctx") or 0)
            needed_tokens = int(meta.get("max_tokens") or 0)
            cached_ctx = int(self._active.meta.get("n_ctx") or 0)
            if needed_ctx > 0 and cached_ctx < needed_ctx:
                return None
            if backend != BackendKind.LLAMA:
                return self._active.handle
            if needed_ctx <= 0 and needed_tokens <= 0:
                # Direct switch() callers have no llama context policy to
                # re-evaluate. The matching active key is a complete cache hit.
                return self._active.handle
            # Reuse a larger cached context/completion budget when headroom is OK.
            # Exact-match used to force reloads on every context-bucket step.
            cached_tokens = int(getattr(self._active.handle, "_seiso_max_tokens", 512) or 512)
            if needed_tokens > 0 and cached_tokens < needed_tokens:
                return None
            cached_layers = int(self._active.meta.get("n_gpu_layers", -1))
            requested_layers = env_int("SEISO_LLAMA_GPU_LAYERS", _mp()._default_llama_gpu_layers())
            if _mp()._llama_cache_is_optimal(
                load_path,
                cached_layers,
                requested_layers,
                n_ctx=needed_ctx or cached_ctx or 2048,
            ) and _mp()._llama_cache_headroom_ok(
                self._active.handle, max_tokens=needed_tokens or cached_tokens
            ):
                return self._active.handle
            return None

    def switch(
        self,
        model_path: str,
        backend: BackendKind,
        loader_fn,
        *,
        cache_key: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        """Load model_path, unloading any previously active model first."""
        norm = self.normalize_path(model_path)
        # Only resolve to an absolute filesystem path when the input is an
        # existing local file/dir.  HuggingFace repo IDs like "org/model-name"
        # must be passed through unchanged so transformers can download them.
        raw = Path(model_path).expanduser()
        load_path = str(raw.absolute()) if raw.exists() else str(model_path)
        key = cache_key or f"{backend.value}:{norm}"
        meta = meta or {}
        cached = self._switch_cache_hit(key, backend, load_path, meta)
        if cached is not None:
            return cached

        with _switch_load_lock:
            cached = self._switch_cache_hit(key, backend, load_path, meta)
            if cached is not None:
                return cached

            if self._active:
                if self._is_same_model_reload(key, backend, norm_path=norm):
                    # Preload/chat often reload the warmed handle (larger n_ctx,
                    # safer batch, layer fit). Preserve generation so the active
                    # chat request is not discarded after reload completes.
                    self._unload_active_immediate()
                else:
                    self._clear_active_for_switch()
                _mp()._clear_optimal_layers_cache()
                _mp()._refresh_headroom_stats(force=True)
            from seiso.memory.protection import (
                ensure_load_fits,
                estimate_path_vram_mb,
                headroom_mb,
                release_cached_memory,
            )

            if backend == BackendKind.LLAMA:
                from seiso.inference.llama_vision import resolve_mmproj_path

                est_mb = int(estimate_path_vram_mb(load_path))
                mmproj = resolve_mmproj_path(load_path)
                if mmproj:
                    est_mb += int(estimate_path_vram_mb(mmproj))
            else:
                est_mb = int(estimate_path_vram_mb(load_path))
            # Interpret this model vs free headroom; free caches if short (non-blocking load prep).
            if est_mb > 0 and headroom_mb() < int(est_mb * 0.98):
                if self._active:
                    self._clear_active_for_switch()
                self._free_memory()
                release_cached_memory(sync=True)
                _mp()._clear_optimal_layers_cache()
                _mp()._refresh_headroom_stats(force=True)

            logger.info("Loading model: %s (%s)", norm, backend.value)
            ensure_load_fits(load_path, mode="chat", backend=backend.value)
            self._ensure_resident_gpu_resource_lock()
            try:
                handle = loader_fn(load_path)
            except Exception:
                self._free_memory()
                self._release_resident_gpu_resource_lock()
                raise
            layer_meta: dict[str, Any] = {}
            if backend == BackendKind.LLAMA:
                layer_meta["n_gpu_layers"] = int(getattr(handle, "_seiso_n_gpu_layers", -1))
                requested_ctx = int((meta or {}).get("n_ctx") or 0)
                layer_meta["n_ctx"] = int(
                    getattr(handle, "_seiso_n_ctx", requested_ctx) or requested_ctx or 0
                )
            elif backend == BackendKind.TORCH:
                torch_model = handle[0] if isinstance(handle, tuple) and handle else handle
                layer_meta["load_precision"] = str(
                    getattr(torch_model, "_seiso_load_precision", "unknown")
                )
                load_policy = getattr(torch_model, "_seiso_load_policy", None)
                if isinstance(load_policy, dict):
                    layer_meta["load_policy"] = dict(load_policy)
            with self._lock:
                if (
                    self._active
                    and self._active.key == key
                    and self._switch_cache_hit(key, backend, load_path, meta) is not None
                ):
                    try:
                        if hasattr(handle, "close"):
                            handle.close()
                    except Exception:
                        logger.debug(
                            "Failed to close duplicate load handle",
                            exc_info=True,
                        )
                    return self._active.handle
                # cancel_and_unload during load must win: discard the fresh handle
                # and keep unload_pending so we do not leave a resident model.
                if self._unload_pending:
                    stale = self._active
                    self._active = None
                    discard = handle
                    handle = None
                else:
                    if self._active:
                        stale = self._active
                        self._active = None
                    else:
                        stale = None
                    discard = None
                    self._active = LoadedModel(
                        key=key,
                        backend=backend,
                        handle=handle,
                        meta={
                            "path": load_path,
                            "norm_path": norm,
                            **(meta or {}),
                            **layer_meta,
                        },
                    )
            if discard is not None:
                # Torch/MLX handles have no .close() — always go through release.
                try:
                    self._release_handle(
                        LoadedModel(
                            key=key,
                            backend=backend,
                            handle=discard,
                            meta={"path": load_path, "norm_path": norm, **(meta or {})},
                        )
                    )
                except Exception:
                    logger.debug(
                        "Failed to release cancelled load handle",
                        exc_info=True,
                    )
                    self._release_resident_gpu_resource_lock()
                if stale is not None:
                    try:
                        self._release_handle(stale)
                    except Exception:
                        logger.debug(
                            "Failed to release stale pool handle", exc_info=True
                        )
                with self._lock:
                    self._unload_pending = False
                clear_dflash_draft_cache()
                raise RuntimeError("Model load cancelled (unload requested)")
            if stale is not None:
                try:
                    self._release_handle(stale)
                except Exception:
                    logger.debug("Failed to release stale pool handle", exc_info=True)
            return handle

    def get_llama(
        self,
        model_path: str,
        n_ctx: int = 4096,
        *,
        tier: str = "normal",
        max_tokens: int = 512,
    ) -> Any:
        def loader(path: str):
            return _mp()._load_llama_model(path, n_ctx, tier=tier, max_tokens=max_tokens)

        norm = self.normalize_path(model_path)
        requested_layers = env_int("SEISO_LLAMA_GPU_LAYERS", _mp()._default_llama_gpu_layers())
        with self._lock:
            if (
                tier == "normal"
                and self._active
                and self._active.backend == BackendKind.LLAMA
                and self._active.meta.get("norm_path") == norm
                and self._active.meta.get("load_tier", "normal") == "normal"
            ):
                cached_ctx = int(self._active.meta.get("n_ctx") or 0)
                cached_layers = int(self._active.meta.get("n_gpu_layers", -1))
                cached_max_tokens = int(
                    getattr(self._active.handle, "_seiso_max_tokens", 512) or 512
                )
                if (
                    cached_ctx >= n_ctx
                    and cached_max_tokens >= max_tokens
                    and _mp()._llama_cache_is_optimal(
                        str(self._active.meta.get("path") or model_path),
                        cached_layers,
                        requested_layers,
                        n_ctx=n_ctx,
                    )
                    and _mp()._llama_cache_headroom_ok(self._active.handle, max_tokens=max_tokens)
                ):
                    return self._active.handle

        key = (
            f"llama:{norm}:tokens:{max_tokens}"
            if tier == "normal"
            else f"llama:{norm}:{tier}:tokens:{max_tokens}"
        )
        return self.switch(
            model_path,
            BackendKind.LLAMA,
            loader,
            cache_key=key,
            meta={"n_ctx": n_ctx, "load_tier": tier, "max_tokens": max_tokens},
        )

    def reload_llama(
        self,
        model_path: str,
        n_ctx: int,
        *,
        tier: str,
        batch_override: tuple[int, int] | None = None,
        max_tokens: int = 512,
    ) -> Any:
        """Unload and reload llama.cpp at a lower memory tier after inference OOM.

        Preserves the active generation id and does not wait on the caller's
        own inference ref (recovery always runs under begin_inference).
        """
        self._unload_active_immediate()
        _mp()._clear_optimal_layers_cache()
        _mp()._refresh_headroom_stats(force=True)
        if batch_override is None:
            return self.get_llama(model_path, n_ctx=n_ctx, tier=tier, max_tokens=max_tokens)

        def loader(path: str):
            return _mp()._load_llama_model(
                path,
                n_ctx,
                tier=tier,
                batch_override=batch_override,
                max_tokens=max_tokens,
            )

        norm = self.normalize_path(model_path)
        key = (
            f"llama:{norm}:{tier}:batch:{batch_override[0]}:{batch_override[1]}:tokens:{max_tokens}"
        )
        return self.switch(
            model_path,
            BackendKind.LLAMA,
            loader,
            cache_key=key,
            meta={"n_ctx": n_ctx, "load_tier": tier, "max_tokens": max_tokens},
        )

    def reload_llama_compact(self, model_path: str, n_ctx: int) -> Any:
        """Backward-compatible alias for compact-tier reload."""
        return self.reload_llama(model_path, n_ctx, tier="compact")

    def get_llamaswap(self, model_path: str, *, num_ctx: int | None = None) -> Any:
        def loader(_path: str):
            from seiso.inference.llamaswap import create_isolated_gguf_client

            client = create_isolated_gguf_client()
            client.ensure_ready()
            return client

        norm = self.normalize_path(model_path)
        key = f"llamaswap:{norm}"
        meta: dict[str, Any] = {"sidecar": True}
        if num_ctx is not None and int(num_ctx) > 0:
            meta["n_ctx"] = int(num_ctx)
        client = self.switch(
            model_path,
            BackendKind.LLAMASWAP,
            loader,
            cache_key=key,
            meta=meta,
        )
        # Refresh pinned ctx on cache hits so preload planning sticks.
        if num_ctx is not None and int(num_ctx) > 0:
            with self._lock:
                if self._active and self._active.key == key:
                    existing = int(self._active.meta.get("n_ctx") or 0)
                    self._active.meta["n_ctx"] = max(existing, int(num_ctx))
                    self._active.meta["sidecar"] = True
        ensure_ready = getattr(client, "ensure_ready", None)
        if callable(ensure_ready):
            ensure_ready()
        return client

    def pinned_n_ctx(self, model_path: str | None = None) -> int | None:
        """Return the active handle's pinned context when it matches ``model_path``."""
        with self._lock:
            active = self._active
            if active is None:
                return None
            raw = active.meta.get("n_ctx")
            if raw is None:
                return None
            try:
                pinned = int(raw)
            except (TypeError, ValueError):
                return None
            if pinned <= 0:
                return None
            if model_path:
                norm = self.normalize_path(model_path)
                active_norm = str(active.meta.get("norm_path") or active.meta.get("path") or "")
                # Key may be llamaswap:/abs/path — also compare key suffix.
                if (
                    active_norm
                    and self.normalize_path(active_norm) != norm
                    and not active.key.endswith(f":{norm}")
                    and active.key != norm
                ):
                    return None
            return pinned

    def get_mlx(self, model_path: str) -> tuple[Any, Any]:
        def loader(path: str):
            from seiso.models.loader import LoadOptions, ModelKind
            from seiso.models.mlx_loader import load_mlx

            return load_mlx(LoadOptions(model_id=path, kind=ModelKind.TEXT))

        return cast(tuple[Any, Any], self.switch(model_path, BackendKind.MLX, loader))

    def get_torch(self, model_path: str, *, load_in_4bit: bool | None = None) -> tuple[Any, Any]:
        from seiso.inference.torch_load_policy import resolve_torch_load_policy
        from seiso.memory.protection import headroom_mb

        norm = self.normalize_path(model_path)
        precision_override = env_str("SEISO_TORCH_LOAD_PRECISION", "auto").strip().lower()
        if load_in_4bit is None and precision_override == "auto":
            with self._lock:
                active = self._active
                if (
                    active is not None
                    and active.backend == BackendKind.TORCH
                    and str(active.meta.get("norm_path") or "") == norm
                    and isinstance(active.handle, tuple)
                ):
                    # Do not re-plan from post-load headroom: doing so can
                    # change BF16 to 4-bit and reload an already-resident model.
                    return cast(tuple[Any, Any], active.handle)

        policy = resolve_torch_load_policy(
            model_path,
            free_mb=int(headroom_mb()),
            force_4bit=load_in_4bit,
        )

        def loader(path: str):
            return self._load_torch_pair(path, load_policy=policy)

        return cast(
            tuple[Any, Any],
            self.switch(
                model_path,
                BackendKind.TORCH,
                loader,
                cache_key=f"torch:{norm}:{policy.precision}",
                meta={"requested_precision": policy.precision},
            ),
        )

    @staticmethod
    def torch_speculative_pair_fits(target_path: str, draft_path: str) -> bool:
        """Return whether both Torch models fit current headroom together."""
        from seiso.memory.protection import estimate_path_vram_mb, headroom_mb

        target_mb = int(estimate_path_vram_mb(target_path, mode="chat"))
        draft_mb = int(estimate_path_vram_mb(draft_path, mode="chat"))
        needed_mb = target_mb + draft_mb
        free_mb = int(headroom_mb())
        return target_mb > 0 and draft_mb > 0 and free_mb > 0 and needed_mb <= free_mb

    def _load_torch_pair(
        self,
        model_path: str,
        *,
        load_in_4bit: bool = True,
        load_policy: Any | None = None,
    ) -> tuple[Any, Any]:
        from seiso.inference.tuning import prepare_torch_model
        from seiso.memory.protection import is_oom_error, release_cached_memory
        from seiso.models.loader import LoadOptions, ModelKind, load_model

        use_4bit = bool(load_policy.load_in_4bit) if load_policy is not None else load_in_4bit
        dtype = load_policy.dtype if load_policy is not None else None

        def load(*, quantized: bool, resolved_dtype: str | None):
            return load_model(
                LoadOptions(
                    model_id=model_path,
                    kind=ModelKind.TEXT,
                    load_in_4bit=quantized,
                    dtype=resolved_dtype,
                    device_map="auto",
                )
            )

        try:
            model, tokenizer = load(quantized=use_4bit, resolved_dtype=dtype)
            actual_precision = (
                "4bit"
                if use_4bit
                else (load_policy.precision if load_policy is not None else (dtype or "native"))
            )
        except Exception as exc:
            text = str(exc).lower()
            can_fallback = not use_4bit and (
                is_oom_error(exc) or "bfloat16" in text or "float16" in text or "dtype" in text
            )
            if not can_fallback:
                raise
            logger.warning(
                "Native half-precision Torch load failed; retrying with 4-bit: %s",
                exc,
            )
            release_cached_memory(sync=True)
            model, tokenizer = load(quantized=True, resolved_dtype=None)
            actual_precision = "4bit-fallback"
        model._seiso_load_precision = actual_precision
        if load_policy is not None:
            model._seiso_load_policy = load_policy.as_dict()
        model = prepare_torch_model(model)
        return model, tokenizer

    def get_torch_speculative(
        self, target_path: str, draft_path: str, *, load_in_4bit: bool = True
    ) -> Any:
        """Load target + draft models for speculative decoding (torch draft)."""
        from seiso.inference.speculative import TorchSpeculativeBundle

        target_norm = self.normalize_path(target_path)
        draft_norm = self.normalize_path(draft_path)
        key = f"spec:{target_norm}:{draft_norm}"

        def loader(_path: str) -> TorchSpeculativeBundle:
            from seiso.memory.protection import (
                ensure_load_fits,
                estimate_path_vram_mb,
                headroom_mb,
            )

            target_mb = int(estimate_path_vram_mb(target_path, mode="chat"))
            draft_mb = int(estimate_path_vram_mb(draft_path, mode="chat"))
            needed_mb = target_mb + draft_mb
            free_mb = headroom_mb()
            if target_mb <= 0 or draft_mb <= 0 or free_mb <= 0:
                raise RuntimeError(
                    "Speculative pair memory could not be safely sized; use target-only generation"
                )
            if needed_mb > free_mb:
                raise RuntimeError(
                    "Speculative pair exceeds free memory: "
                    f"needs ~{needed_mb}MB (target={target_mb}MB + "
                    f"draft={draft_mb}MB), free={free_mb}MB"
                )
            ensure_load_fits(target_path, mode="chat", backend=BackendKind.TORCH.value)
            ensure_load_fits(draft_path, mode="chat", backend=BackendKind.TORCH.value)
            target_model, target_tokenizer = self._load_torch_pair(
                target_path, load_in_4bit=load_in_4bit
            )
            draft_model, draft_tokenizer = self._load_torch_pair(
                draft_path, load_in_4bit=load_in_4bit
            )
            return TorchSpeculativeBundle(
                target_model=target_model,
                target_tokenizer=target_tokenizer,
                draft_model=draft_model,
                draft_tokenizer=draft_tokenizer,
            )

        return self.switch(
            target_path,
            BackendKind.TORCH,
            loader,
            cache_key=key,
            meta={
                "path": target_path,
                "norm_path": target_norm,
                "draft_path": draft_path,
                "draft_norm_path": draft_norm,
            },
        )

    # Note: dflash drafts are loaded directly in the runner using llama_cpp.Llama
    # to avoid interfering with the primary target model's active handle in the pool.

    def unload_all(self) -> None:
        """Release all loaded models and clear GPU memory."""
        with self._lock:
            if self._inference_refs > 0:
                self._unload_pending = True
                return
            active = self._active
            self._active = None
            self._unload_pending = False

        if active is not None:
            self._release_handle(active)
        else:
            clear_dflash_draft_cache()
            # Pool empty — still ask sidecars to drop orphan residency.
            self._release_orphan_sidecars()

    def _release_orphan_sidecars(self) -> None:
        """Best-effort unload of Ollama/llama-swap models not tracked in the pool."""
        try:
            from seiso.inference.llamaswap import release_orphan_sidecar_memory

            notes = release_orphan_sidecar_memory()
        except Exception:
            logger.debug("Orphan sidecar unload skipped", exc_info=True)
            return
        if not notes:
            return
        with self._lock:
            self._release_notes.extend(notes)

    def _free_memory(self, *, sync: bool = False) -> None:
        from seiso.memory.protection import release_cached_memory

        release_cached_memory(sync=sync)
        _mp()._clear_optimal_layers_cache()

    def status(self) -> dict:
        with self._lock:
            active = self._active
            n_ctx = None
            if active is not None:
                raw = active.meta.get("n_ctx")
                if raw is not None:
                    try:
                        n_ctx = int(raw)
                    except (TypeError, ValueError):
                        n_ctx = None
            return {
                "active_model": active.key if active else None,
                "backend": active.backend.value if active else None,
                "path": active.meta.get("path") if active else None,
                "draft_path": active.meta.get("draft_path") if active else None,
                "n_ctx": n_ctx,
                "load_precision": (active.meta.get("load_precision") if active else None),
                "load_policy": active.meta.get("load_policy") if active else None,
                "release_notes": list(self._release_notes),
                "inference_refs": self._inference_refs,
                "unload_pending": self._unload_pending,
            }


def get_model_pool() -> ModelPool:
    return ModelPool.get()
