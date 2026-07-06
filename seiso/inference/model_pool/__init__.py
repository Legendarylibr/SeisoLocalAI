"""VRAM-managed model pool — unloads previous model when switching."""

from __future__ import annotations

from seiso.inference.model_pool import dflash, llama_load, pool


_MUTABLE_LLAMA_STATE = frozenset({"_llama_offload_checked", "_llama_offload_supported"})


def _reexport(module) -> None:
    for name in dir(module):
        if name.startswith("__") or name in _MUTABLE_LLAMA_STATE:
            continue
        globals()[name] = getattr(module, name)


_reexport(llama_load)
_reexport(dflash)
_reexport(pool)


def __getattr__(name: str):
    if name in {"_llama_offload_checked", "_llama_offload_supported"}:
        return getattr(llama_load, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    name
    for name in globals()
    if not name.startswith("__") and name not in {"dflash", "llama_load", "pool", "_reexport"}
]
