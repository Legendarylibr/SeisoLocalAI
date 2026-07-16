"""Online rollout backends for slime-style GRPO.

* ``data_gen`` — colocated Hugging Face ``generate`` (default; single-GPU path)
* ``sglang`` — OpenAI-compatible HTTP generation against a running SGLang server
  (preferred for multi-GPU / high-throughput generation)

Completions are always produced online. Prompt corpora (labels/tests) come from
JSONL or high-level ``data_gen`` corpus materialization — never from stored
model outputs.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from seiso.slime_single_gpu.config import SingleGpuSlimeConfig

# Canonical: hf | sglang | auto. "data_gen" is a legacy alias of "hf".
_ROLLOUT_BACKENDS = frozenset({"hf", "sglang", "auto", "data_gen"})


def _normalize_backend_name(name: str) -> str:
    key = str(name or "hf").lower().strip()
    if key == "data_gen":
        return "hf"
    return key


def resolve_rollout_backend(
    config: SingleGpuSlimeConfig,
    *,
    world_size: int = 1,
) -> str:
    """Resolve effective backend.

    * ``hf`` (default; alias ``data_gen``) — colocated Hugging Face generate
    * ``sglang`` — OpenAI-compatible SGLang HTTP
    * ``auto`` — SGLang when ``sglang_base_url`` is set and ``world_size > 1``
    """
    name = _normalize_backend_name(
        getattr(config, "rollout_backend", "hf") or "hf"
    )
    if name not in {"hf", "sglang", "auto"}:
        raise ValueError(
            f"rollout_backend must be one of: hf, sglang, auto (got {name!r})"
        )
    if name == "auto":
        base = str(getattr(config, "sglang_base_url", "") or "").strip()
        if base and world_size > 1:
            return "sglang"
        return "hf"
    return name


def validate_rollout_backend_config(config: SingleGpuSlimeConfig) -> None:
    name = _normalize_backend_name(
        getattr(config, "rollout_backend", "hf") or "hf"
    )
    if name not in {"hf", "sglang", "auto"}:
        raise ValueError(
            f"rollout_backend must be one of: hf, sglang, auto (got {name!r})"
        )
    if name == "sglang":
        base = str(getattr(config, "sglang_base_url", "") or "").strip()
        if not base:
            raise ValueError(
                "rollout_backend=sglang requires sglang_base_url "
                "(e.g. http://127.0.0.1:30000)"
            )
    timeout = float(getattr(config, "sglang_timeout_s", 120.0) or 120.0)
    if timeout <= 0:
        raise ValueError("sglang_timeout_s must be positive")
    max_workers = int(getattr(config, "sglang_max_workers", 8) or 8)
    if max_workers < 1:
        raise ValueError("sglang_max_workers must be positive")


@dataclass(frozen=True)
class GeneratedChunk:
    """One generation chunk: parallel lists aligned as prompt_idx * n + k."""

    prompts: list[str]
    completions: list[str]
    # When set (data_gen), full sequences from HF generate; else None → re-tokenize.
    sequences: Any | None = None
    prompt_width: int | None = None


def _as_chat_messages(prompt: str | list[Any]) -> list[dict[str, str]]:
    """Normalize slime chat prompts or plain strings to OpenAI-style messages."""
    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for item in prompt:
            if isinstance(item, dict) and "content" in item:
                role = str(item.get("role") or "user")
                messages.append({"role": role, "content": str(item["content"])})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages or [{"role": "user", "content": ""}]
    return [{"role": "user", "content": str(prompt)}]


def format_generation_prompt(
    tokenizer,
    prompt: str | list[Any],
    config: SingleGpuSlimeConfig,
) -> str:
    """Apply optional chat template (slime --apply-chat-template), then thinking open."""
    messages = _as_chat_messages(prompt)
    # Optional thinking instruction on the last user turn only.
    if config.require_thinking_trace:
        last = messages[-1]
        content = last["content"]
        if "<think>" not in content.lower():
            last = {
                **last,
                "content": (
                    f"{content.rstrip()}\n\n{config.thinking_instruction}\n<think>"
                ),
            }
            messages = [*messages[:-1], last]

    use_chat = bool(getattr(config, "apply_chat_template", True))
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    # Fallback: concatenate message contents (no template).
    return "\n".join(m["content"] for m in messages)


def generate_data_gen_chunk(
    *,
    generation_model,
    tokenizer,
    prompts: list[str],
    config: SingleGpuSlimeConfig,
    torch,
) -> GeneratedChunk:
    """Colocated HF generate — preserves prior single-GPU rollout behavior."""
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_tokens,
    ).to(config.device)
    prompt_width = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        generated = generation_model.generate(
            **encoded,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_return_sequences=config.rollouts_per_prompt,
        )
    completions = tokenizer.batch_decode(
        generated[:, prompt_width:],
        skip_special_tokens=True,
    )
    # Expand prompts to match num_return_sequences layout.
    expanded_prompts: list[str] = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * config.rollouts_per_prompt)
    return GeneratedChunk(
        prompts=expanded_prompts,
        completions=list(completions),
        sequences=generated,
        prompt_width=prompt_width,
    )


def generate_sglang_chunk(
    *,
    tokenizer,
    prompts: list[str],
    config: SingleGpuSlimeConfig,
) -> GeneratedChunk:
    """Generate ``rollouts_per_prompt`` completions per prompt via SGLang HTTP."""
    n = config.rollouts_per_prompt
    client = SGLangRolloutClient.from_config(config)
    # Expand to one request per (prompt, sample) for broad API compatibility.
    jobs: list[tuple[int, str]] = []
    for p_idx, prompt in enumerate(prompts):
        for _ in range(n):
            jobs.append((p_idx, prompt))

    results: list[str | None] = [None] * len(jobs)
    max_workers = min(
        int(getattr(config, "sglang_max_workers", 8) or 8),
        max(1, len(jobs)),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(client.complete, prompt): idx
            for idx, (_p_idx, prompt) in enumerate(jobs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    completions = [text if text is not None else "" for text in results]
    expanded_prompts = [prompt for _p_idx, prompt in jobs]
    return GeneratedChunk(
        prompts=expanded_prompts,
        completions=completions,
        sequences=None,
        prompt_width=None,
    )


def build_sequence_tensors(
    *,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    config: SingleGpuSlimeConfig,
    torch,
    device: str,
) -> list[dict[str, Any]]:
    """Tokenize prompt+completion pairs into per-row rollout tensors.

    Note: external SGLang generation is text-level; retokenizing completions can
    diverge slightly from the engine's BPE. Prefer HF ``generate`` sequences for
    exact on-policy tokens. We tokenize the *same prompt string* the server saw
    and append completion tokens without special tokens on the response.
    """
    rows: list[dict[str, Any]] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for prompt, completion in zip(prompts, completions, strict=True):
        # Match the prompt bytes already sent to SGLang (already chat-templated).
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=config.max_prompt_tokens,
            return_tensors="pt",
        )["input_ids"][0]
        if completion:
            comp_ids = tokenizer(
                completion,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_new_tokens,
                return_tensors="pt",
            )["input_ids"][0]
        else:
            comp_ids = torch.zeros(0, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, comp_ids], dim=0).to(device)
        attention_mask = torch.ones_like(input_ids, device=device)
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        prompt_len = int(prompt_ids.numel())
        response_mask[prompt_len:] = True
        if pad_id is not None:
            response_mask = response_mask & (input_ids != pad_id)
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "prompt_len": prompt_len,
            }
        )
    return rows


class SGLangRolloutClient:
    """Minimal OpenAI-compatible client for SGLang ``/v1/completions``."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 120.0,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_tokens: int = 256,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    @classmethod
    def from_config(cls, config: SingleGpuSlimeConfig) -> SGLangRolloutClient:
        engines = sglang_engine_urls(config)
        if not engines:
            raise ValueError("sglang_base_url is required for SGLang rollout")
        # Generation uses the first engine; weight sync fans out to all URLs.
        base = engines[0]
        _validate_sglang_url(base)
        model = str(getattr(config, "sglang_model", "") or "").strip() or config.model_id
        return cls(
            base_url=base,
            model=model,
            api_key=str(getattr(config, "sglang_api_key", "EMPTY") or "EMPTY"),
            timeout_s=float(getattr(config, "sglang_timeout_s", 120.0) or 120.0),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            max_tokens=int(config.max_new_tokens),
        )

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
        }
        data = self._post_json("/v1/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("SGLang /v1/completions returned no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("SGLang choice payload is invalid")
        text = first.get("text")
        if not isinstance(text, str):
            # Some servers put chat-style content here.
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            raise RuntimeError("SGLang choice missing text")
        return text

    def update_weights_from_disk(
        self,
        model_path: str,
        *,
        weight_version: str | None = None,
    ) -> dict[str, Any]:
        """Hot-reload HF weights (slime / SGLang ``update_weights_from_disk``)."""
        payload: dict[str, Any] = {"model_path": model_path}
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._post_json("/update_weights_from_disk", payload)

    def pull_weights(
        self,
        *,
        local_checkpoint_dir: str,
        source_dir: str,
        target_version: int,
    ) -> dict[str, Any]:
        """Slime-patched engines: apply disk delta then local reload.

        Vanilla SGLang does not implement this endpoint — callers must fall back
        to ``update_weights_from_disk`` with a full HF checkpoint.
        """
        return self._post_json(
            "/pull_weights",
            {
                "local_checkpoint_dir": local_checkpoint_dir,
                "source_dir": source_dir,
                "target_version": int(target_version),
            },
        )

    def flush_cache(self) -> dict[str, Any] | None:
        """Best-effort KV cache flush after a weight update."""
        try:
            return self._request("GET", "/flush_cache", body=None)
        except RuntimeError:
            return None

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body=payload)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"SGLang HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"SGLang request failed: {exc}") from exc
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}


def sglang_engine_urls(config: SingleGpuSlimeConfig) -> list[str]:
    """Resolve one or more SGLang engine base URLs (comma-separated or multi field)."""
    urls: list[str] = []
    primary = str(getattr(config, "sglang_base_url", "") or "").strip()
    if primary:
        urls.extend(part.strip() for part in primary.split(",") if part.strip())
    extra = getattr(config, "sglang_engine_urls", None) or []
    if isinstance(extra, str):
        urls.extend(part.strip() for part in extra.split(",") if part.strip())
    elif isinstance(extra, (list, tuple)):
        urls.extend(str(u).strip() for u in extra if str(u).strip())
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        key = url.rstrip("/")
        _validate_sglang_url(key)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _validate_sglang_url(url: str) -> None:
    """Reject non-HTTP(S) schemes (basic SSRF hardening for config-controlled URLs)."""
    lowered = url.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError(
            f"sglang URL must use http:// or https:// scheme, got {url!r}"
        )


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
    partial.rename(final)
    return str(final.resolve())


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
        raise RuntimeError(
            "SGLang weight sync failed on one or more engines: " + "; ".join(errors)
        )


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
    _broadcast_weights_to_engines(
        config, model_path=used_path, weight_version=version
    )
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
