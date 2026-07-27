"""Actor weight export and SGLang/vLLM hot-reload for slime rollouts."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.rollout_clients import SGLangRolloutClient, VLLMRolloutClient
from seiso.slime.rollout_http import resolve_vllm_base_url, sglang_engine_urls, vllm_engine_urls

# resolve_rollout_backend imported lazily / from resolve module
from seiso.slime.rollout_resolve import resolve_rollout_backend


def export_actor_checkpoint(model, tokenizer, output_dir: Any) -> str:
    """Write a full HF checkpoint suitable for SGLang disk weight reload.

    Writes to ``*.partial`` then atomically renames so engines never load a
    half-written tree. For PEFT/LoRA, merges adapters for export only.
    """
    from pathlib import Path

    final = Path(output_dir)
    partial = final.parent / f"{final.name}.partial"
    if partial.exists():
        _rm_tree(partial)
    partial.mkdir(parents=True, exist_ok=True)
    unwrapped = getattr(model, "module", model)

    merged = False
    if hasattr(unwrapped, "merge_adapter"):
        unwrapped.merge_adapter()
        merged = True
    try:
        to_save = unwrapped
        if hasattr(unwrapped, "get_base_model"):
            to_save = unwrapped.get_base_model()
        to_save.save_pretrained(partial)
        tokenizer.save_pretrained(partial)
    finally:
        if merged and hasattr(unwrapped, "unmerge_adapter"):
            unwrapped.unmerge_adapter()
    if final.exists():
        _rm_tree(final)
    # ``replace`` overwrites atomically; ``rename`` can fail if ``final`` races back.
    partial.replace(final)
    return str(final.resolve())


def export_actor_lora_adapter(model, tokenizer, output_dir: Any) -> str:
    """Write PEFT/LoRA adapter weights for vLLM dynamic ``load_lora_adapter``.

    Does **not** merge adapters. Atomic ``*.partial`` rename like full export.
    """
    from pathlib import Path

    final = Path(output_dir)
    partial = final.parent / f"{final.name}.partial"
    if partial.exists():
        _rm_tree(partial)
    partial.mkdir(parents=True, exist_ok=True)
    unwrapped = getattr(model, "module", model)
    # Prefer PEFT adapter export; fall back to full save for non-PEFT actors.
    if hasattr(unwrapped, "save_pretrained") and (
        hasattr(unwrapped, "peft_config")
        or hasattr(unwrapped, "get_base_model")
        or hasattr(unwrapped, "merge_adapter")
    ):
        unwrapped.save_pretrained(partial)
    else:
        unwrapped.save_pretrained(partial)
    tokenizer.save_pretrained(partial)
    if final.exists():
        _rm_tree(final)
    partial.replace(final)
    return str(final.resolve())


def _model_has_lora_adapters(model) -> bool:
    unwrapped = getattr(model, "module", model)
    return bool(
        hasattr(unwrapped, "peft_config")
        or hasattr(unwrapped, "get_base_model")
        or hasattr(unwrapped, "merge_adapter")
    )


def _resolve_vllm_weight_mode(config: SingleGpuSlimeConfig, model) -> str:
    mode = str(getattr(config, "vllm_weight_mode", "auto") or "auto").lower()
    if mode == "auto":
        if bool(getattr(config, "use_lora", False)) or _model_has_lora_adapters(model):
            return "lora"
        return "full"
    return mode


def _rm_tree(path: Any) -> None:
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        return
    for child in sorted(root.rglob("*"), reverse=True):
        with contextlib.suppress(OSError):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    with contextlib.suppress(OSError):
        root.rmdir()


def _actor_state_cpu(model) -> dict[str, Any]:
    """CPU snapshot of actor weights (merged view for PEFT when possible)."""
    unwrapped = getattr(model, "module", model)
    merged = False
    if hasattr(unwrapped, "merge_adapter"):
        unwrapped.merge_adapter()
        merged = True
    try:
        target = unwrapped
        if hasattr(unwrapped, "get_base_model"):
            target = unwrapped.get_base_model()
        state = {
            name: tensor.detach().float().cpu().clone()
            for name, tensor in target.state_dict().items()
        }
    finally:
        if merged and hasattr(unwrapped, "unmerge_adapter"):
            unwrapped.unmerge_adapter()
    return state


def _write_delta_payload(
    *,
    prev: dict[str, Any],
    curr: dict[str, Any],
    out_dir: Any,
    step: int,
) -> tuple[str, int, int]:
    """Write changed tensors as safetensors delta (slime-style disk delta)."""
    from pathlib import Path

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    changed: dict[str, Any] = {}
    for name, tensor in curr.items():
        old = prev.get(name)
        if old is None or old.shape != tensor.shape or not bool((old == tensor).all().item()):
            changed[name] = tensor
    meta = {
        "format": "seiso_sglang_delta_v1",
        "step": int(step),
        "num_tensors_total": len(curr),
        "num_tensors_changed": len(changed),
        "tensor_names": sorted(changed),
    }
    (path / "delta_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if changed:
        try:
            from safetensors.torch import save_file

            save_file(changed, str(path / "delta.safetensors"))
        except Exception:
            # Fallback without safetensors dependency shape
            import torch

            torch.save(changed, path / "delta.pt")
    return str(path.resolve()), len(changed), len(curr)


def _prune_weight_versions(weight_root: Any, keep: int) -> None:
    """Prune both full ``weight_v*`` and delta ``delta_v*`` version dirs."""
    from pathlib import Path

    root = Path(weight_root)
    if not root.is_dir() or keep < 1:
        return
    for prefix in ("weight_v", "delta_v"):
        versions = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
            key=lambda p: p.name,
        )
        for old in versions[:-keep]:
            _rm_tree(old)


class WeightSyncState:
    """Mutable rank0 state for full/delta SGLang weight transport."""

    def __init__(self) -> None:
        self.prev_state: dict[str, Any] | None = None
        self.last_full_path: str | None = None
        self.last_mode: str | None = None
        self.last_changed: int = 0


def _broadcast_weights_to_engines(
    config: SingleGpuSlimeConfig,
    *,
    model_path: str,
    weight_version: str,
) -> None:
    engines = sglang_engine_urls(config)
    if not engines:
        raise ValueError("sglang_base_url is required for SGLang weight sync")
    errors: list[str] = []
    for base in engines:
        client = SGLangRolloutClient.from_config(config)
        client.base_url = base
        try:
            client.update_weights_from_disk(model_path, weight_version=weight_version)
            client.flush_cache()
        except RuntimeError as exc:
            errors.append(f"{base}: {exc}")
    if errors:
        raise RuntimeError("SGLang weight sync failed on one or more engines: " + "; ".join(errors))


def sync_sglang_weights_from_actor(
    *,
    model,
    tokenizer,
    config: SingleGpuSlimeConfig,
    step: int,
    is_main: bool,
    active_backend: str | None = None,
    sync_state: WeightSyncState | None = None,
) -> str | None:
    """Rank-0 export + multi-engine SGLang hot-reload (slime disk transport).

    Modes (``sglang_weight_mode``):
    * ``full`` — always write a complete HF checkpoint + ``update_weights_from_disk``
    * ``delta`` — skip if no tensors changed; else try slime ``/pull_weights`` with a
      safetensors delta, falling back to full disk reload on vanilla SGLang

    Returns the checkpoint path when a sync ran on this process, else None.
    Non-main ranks return None (caller must barrier).
    """
    if not bool(getattr(config, "sglang_sync_weights", True)):
        return None
    backend = active_backend or resolve_rollout_backend(config, world_size=1)
    if backend != "sglang":
        return None
    if not is_main:
        return None

    mode = str(getattr(config, "sglang_weight_mode", "full") or "full").lower()
    if mode not in {"full", "delta"}:
        raise ValueError("sglang_weight_mode must be 'full' or 'delta'")

    weight_root = config.output_dir / str(
        getattr(config, "sglang_weight_dir", "sglang_weight_sync") or "sglang_weight_sync"
    )
    weight_root.mkdir(parents=True, exist_ok=True)
    version = f"v{int(step)}"
    ckpt_path = weight_root / f"weight_v{int(step):06d}"
    state = sync_state or WeightSyncState()
    keep = int(getattr(config, "sglang_weight_keep", 2) or 2)

    # ---- delta: snapshot only when needed (avoid full float clone on full mode) ----
    if mode == "delta" and state.prev_state is not None:
        curr = _actor_state_cpu(model)
        delta_dir = weight_root / f"delta_v{int(step):06d}"
        delta_path, n_changed, _n_total = _write_delta_payload(
            prev=state.prev_state,
            curr=curr,
            out_dir=delta_dir,
            step=step,
        )
        state.last_changed = n_changed
        if n_changed == 0:
            state.last_mode = "delta_skip"
            _prune_weight_versions(weight_root, keep)
            return state.last_full_path or delta_path

        # Try slime-patched /pull_weights on each engine (apply delta → local full).
        if state.last_full_path:
            pull_ok = True
            for base in sglang_engine_urls(config):
                client = SGLangRolloutClient.from_config(config)
                client.base_url = base
                try:
                    client.pull_weights(
                        local_checkpoint_dir=state.last_full_path,
                        source_dir=delta_path,
                        target_version=int(step),
                    )
                except RuntimeError:
                    pull_ok = False
                    break
            if pull_ok:
                for base in sglang_engine_urls(config):
                    client = SGLangRolloutClient.from_config(config)
                    client.base_url = base
                    client.flush_cache()
                state.prev_state = curr
                state.last_mode = "delta"
                _prune_weight_versions(weight_root, keep)
                return delta_path
        # Vanilla SGLang: fall through to full checkpoint.

    # ---- full HF checkpoint (default + delta fallback) ----
    used_path = export_actor_checkpoint(model, tokenizer, ckpt_path)
    _broadcast_weights_to_engines(config, model_path=used_path, weight_version=version)
    # Keep CPU snapshot only for future delta diffs (not for full-only runs).
    if mode == "delta":
        state.prev_state = _actor_state_cpu(model)
    else:
        state.prev_state = None
    state.last_full_path = used_path
    state.last_mode = "full"
    state.last_changed = 0
    _prune_weight_versions(weight_root, keep)
    return used_path


def _broadcast_vllm_lora(
    config: SingleGpuSlimeConfig,
    *,
    lora_path: str,
    lora_name: str,
) -> None:
    engines = vllm_engine_urls(config, allow_empty_primary=True)
    if not engines:
        base = resolve_vllm_base_url(config)
        engines = [base] if base else []
    if not engines:
        raise ValueError("vllm_base_url is required for vLLM weight sync")
    errors: list[str] = []
    for base in engines:
        client = VLLMRolloutClient.from_config(config)
        client.base_url = base
        try:
            client.load_lora_adapter(lora_path, lora_name=lora_name)
        except RuntimeError as exc:
            errors.append(f"{base}: {exc}")
    if errors:
        raise RuntimeError(
            "vLLM LoRA weight sync failed on one or more engines: " + "; ".join(errors)
        )


def _broadcast_vllm_full(
    config: SingleGpuSlimeConfig,
    *,
    model_path: str,
    weight_version: str,
) -> None:
    engines = vllm_engine_urls(config, allow_empty_primary=True)
    if not engines:
        base = resolve_vllm_base_url(config)
        engines = [base] if base else []
    if not engines:
        raise ValueError("vllm_base_url is required for vLLM weight sync")
    errors: list[str] = []
    for base in engines:
        client = VLLMRolloutClient.from_config(config)
        client.base_url = base
        try:
            client.pause()
            try:
                client.update_weights_from_disk(
                    model_path, weight_version=weight_version
                )
            finally:
                # Always resume — a failed update must not leave the engine paused.
                try:
                    client.resume()
                except RuntimeError as resume_exc:
                    errors.append(f"{base} (resume): {resume_exc}")
        except RuntimeError as exc:
            errors.append(f"{base}: {exc}")
    if errors:
        raise RuntimeError(
            "vLLM full weight sync failed on one or more engines: "
            + "; ".join(errors)
            + ". Prefer slime_use_lora + vllm_weight_mode=lora (dynamic "
            "/v1/load_lora_adapter), or restart the vLLM server from the exported "
            "checkpoint path."
        )


def sync_vllm_weights_from_actor(
    *,
    model,
    tokenizer,
    config: SingleGpuSlimeConfig,
    step: int,
    is_main: bool,
    active_backend: str | None = None,
    sync_state: WeightSyncState | None = None,
) -> str | None:
    """Rank-0 export + multi-engine vLLM hot-reload for multi-GPU slime rollouts.

    Modes (``vllm_weight_mode``):
    * ``auto`` — LoRA when the actor has PEFT adapters, else full
    * ``lora`` — export PEFT adapter + ``/v1/load_lora_adapter`` (preferred)
    * ``full`` — export merged HF checkpoint + best-effort disk reload endpoints

    Returns the checkpoint/adapter path when a sync ran on this process, else None.
    Non-main ranks return None (caller must barrier).
    """
    if not bool(getattr(config, "vllm_sync_weights", True)):
        return None
    backend = active_backend or resolve_rollout_backend(config, world_size=1)
    if backend != "vllm":
        return None
    if not is_main:
        return None

    mode = _resolve_vllm_weight_mode(config, model)
    weight_root = config.output_dir / str(
        getattr(config, "vllm_weight_dir", "vllm_weight_sync") or "vllm_weight_sync"
    )
    weight_root.mkdir(parents=True, exist_ok=True)
    version = f"v{int(step)}"
    keep = int(getattr(config, "vllm_weight_keep", 2) or 2)
    state = sync_state or WeightSyncState()
    lora_name = (
        str(getattr(config, "vllm_lora_name", "") or "seiso_slime_policy").strip()
        or "seiso_slime_policy"
    )

    if mode == "lora":
        adapter_path = weight_root / f"lora_v{int(step):06d}"
        used_path = export_actor_lora_adapter(model, tokenizer, adapter_path)
        _broadcast_vllm_lora(config, lora_path=used_path, lora_name=lora_name)
        state.last_full_path = used_path
        state.last_mode = "lora"
        state.last_changed = 0
        # Prune both lora_v* and weight_v* trees under the same root.
        _prune_weight_versions(weight_root, keep)
        _prune_named_versions(weight_root, prefix="lora_v", keep=keep)
        return used_path

    ckpt_path = weight_root / f"weight_v{int(step):06d}"
    used_path = export_actor_checkpoint(model, tokenizer, ckpt_path)
    _broadcast_vllm_full(config, model_path=used_path, weight_version=version)
    state.last_full_path = used_path
    state.last_mode = "full"
    state.last_changed = 0
    _prune_weight_versions(weight_root, keep)
    return used_path


def _prune_named_versions(weight_root: Any, *, prefix: str, keep: int) -> None:
    from pathlib import Path

    root = Path(weight_root)
    if not root.is_dir() or keep < 1:
        return
    versions = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.name,
    )
    for old in versions[:-keep]:
        _rm_tree(old)
