"""Patch model modules with leak-safe fused kernels during training."""

from __future__ import annotations

import logging
import types
from collections.abc import Callable
from typing import Any

from seiso.kernels.dispatch import (
    active_backend,
    estimate_vram_savings_pct,
    fused_lora_delta,
    fused_lora_qkv_delta,
    fused_mlp_swiglu,
    fused_rms_norm,
    fused_swiglu,
    kernel_metadata,
)
from seiso.kernels.lifecycle import register_patch, restore_kernel_patches
from seiso.kernels.memory_mode import (
    apply_low_vram_kernel_tuning,
    kernel_low_vram_enabled,
)
from seiso.kernels.platform import detect_gpu

logger = logging.getLogger(__name__)


def _use_fused_cuda_kernels(x: Any) -> bool:
    """Use fused CUDA path on CUDA tensors (autograd wrappers handle training grads)."""
    if not getattr(x, "is_cuda", False):
        return False
    return active_backend() == "cuda"


_RMSNORM_CLASSES = frozenset(
    # GemmaRMSNorm uses (1+weight) — do not patch with Llama-style rms*weight.
    {"LlamaRMSNorm", "Qwen2RMSNorm", "RMSNorm"}
)
_MLP_CLASSES = frozenset(
    {
        "LlamaMLP",
        "MistralMLP",
        "Qwen2MLP",
        "Qwen3MLP",
        # Phi3MLP uses fused gate_up_proj — not gate_proj/up_proj.
        # GemmaMLP / Gemma2MLP use GELU — not SwiGLU/SiLU.
        "MixtralMLP",
    }
)
_DECODER_LAYER_CLASSES = frozenset(
    {
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "Qwen3DecoderLayer",
        # Gemma / Phi-3 use different norm/MLP contracts — leave unpatched.
    }
)
_FUSED_RESIDUAL_DECODER_CLASSES = frozenset(
    {
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "Qwen3DecoderLayer",
    }
)
_QKV_PROJECTION_SUFFIXES = (".q_proj", ".k_proj", ".v_proj")
_ATTENTION_CLASSES = frozenset(
    {
        "LlamaAttention",
        "MistralAttention",
        "Qwen2Attention",
        "Qwen3Attention",
        "GemmaAttention",
        "Gemma2Attention",
        "Phi3Attention",
    }
)


def clear_kernel_patches(model: Any | None = None) -> None:
    """Restore original forwards and release patch registry."""
    restore_kernel_patches(model)


def _patch_forward(model: Any, module: Any, forward_fn: Callable) -> None:
    if hasattr(module, "_seiso_orig_forward"):
        return
    module._seiso_orig_forward = module.forward
    try:
        module.forward = types.MethodType(forward_fn, module)
        register_patch(model, module)
    except Exception:
        module.forward = module._seiso_orig_forward
        delattr(module, "_seiso_orig_forward")
        raise


def _is_swiglu_mlp(module: Any) -> bool:
    """True only for SwiGLU MLPs with separate gate_proj / up_proj / down_proj."""
    if not all(hasattr(module, a) for a in ("gate_proj", "up_proj", "down_proj")):
        return False
    cls = type(module).__name__
    if "Moe" in cls or "MoE" in cls or "Sparse" in cls:
        return False
    act = getattr(module, "act_fn", None)
    if act is None:
        # Known SwiGLU class names are a hint only when attrs already matched.
        return cls in _MLP_CLASSES
    act_name = type(act).__name__.lower()
    return "silu" in act_name or "swish" in act_name


def apply_training_kernels(
    model: Any,
    *,
    use_cuda: bool = True,
    use_triton: bool = True,
    patch_mlp: bool = True,
    low_vram: bool | None = None,
) -> dict[str, Any]:
    """
    Patch RMSNorm and SwiGLU MLP modules with fused GPU kernels.

    NVIDIA: native CUDA. AMD ROCm: Triton. Patches are always restored on cleanup.
    """
    if low_vram is None:
        low_vram = kernel_low_vram_enabled()
    if low_vram:
        apply_low_vram_kernel_tuning()

    platform = detect_gpu()
    meta = kernel_metadata()
    meta.update(
        {
            "cuda_applied": False,
            "triton_applied": False,
            "rmsnorm_patched": 0,
            "mlp_patched": 0,
            "modules_patched": 0,
            "fused_enabled": use_cuda or use_triton,
            "patch_mlp": patch_mlp,
            "kernel_low_vram": low_vram,
        }
    )

    if not (use_cuda or use_triton):
        logger.info("Fused kernels disabled by config")
        return meta

    if platform.device_count == 0:
        logger.info("Fused kernels skipped (no GPU)")
        return meta

    if platform.vendor.value == "amd" and not use_triton:
        logger.info("AMD GPU detected — enable use_triton for fused kernels")
        return meta

    try:
        import torch  # noqa: F401
    except ImportError:
        return meta

    rms_patched = 0
    mlp_patched = 0
    backend = meta["kernel_backend"]

    for _name, module in model.named_modules():
        cls = type(module).__name__

        if cls in _RMSNORM_CLASSES and hasattr(module, "weight"):
            if hasattr(module, "_seiso_orig_forward"):
                continue
            eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))

            def _rms_forward(self, hidden_states, _eps=eps):
                if isinstance(hidden_states, tuple) and len(hidden_states) == 1:
                    hidden_states = hidden_states[0]
                if _use_fused_cuda_kernels(hidden_states):
                    return fused_rms_norm(hidden_states, self.weight, eps=_eps)
                return self._seiso_orig_forward(hidden_states)

            _patch_forward(model, module, _rms_forward)
            rms_patched += 1

        if patch_mlp and _is_swiglu_mlp(module):
            if hasattr(module, "_seiso_orig_forward"):
                continue

            def _mlp_forward(self, hidden_states):
                # Production MLP: module linears (cuBLAS) + fused SwiGLU epilogue.
                # fused_mlp_swiglu also uses torch GEMM + fused_swiglu — same quality;
                # prefer gate_proj/up_proj so LoRA/quant wrappers stay correct.
                if hidden_states.is_cuda and _use_fused_cuda_kernels(hidden_states):
                    if (
                        not _is_peft_lora_linear(self.gate_proj)
                        and not _is_peft_lora_linear(self.up_proj)
                        and _supports_einsum_batch((self.gate_proj, self.up_proj))
                    ):
                        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                        inter = fused_mlp_swiglu(
                            flat,
                            self.gate_proj.weight,
                            self.up_proj.weight,
                        )
                        inter = inter.reshape(*hidden_states.shape[:-1], inter.shape[-1])
                        return self.down_proj(inter)
                    gate = self.gate_proj(hidden_states)
                    up = self.up_proj(hidden_states)
                    return self.down_proj(fused_swiglu(gate, up))
                return self._seiso_orig_forward(hidden_states)

            _patch_forward(model, module, _mlp_forward)
            mlp_patched += 1

    total = rms_patched + mlp_patched
    meta["rmsnorm_patched"] = rms_patched
    meta["mlp_patched"] = mlp_patched
    meta["modules_patched"] = total
    meta["cuda_applied"] = backend == "cuda" and total > 0
    meta["triton_applied"] = backend == "triton" and total > 0
    meta["estimated_vram_savings_pct"] = estimate_vram_savings_pct(
        total > 0, False, low_vram=low_vram
    )

    if total:
        logger.info(
            "Fused kernels: %d RMSNorm + %d MLP | vendor=%s backend=%s device=%s",
            rms_patched,
            mlp_patched,
            platform.vendor.value,
            backend,
            platform.device_name,
        )
    return meta


def _lora_dropout_p(dropout: Any) -> float:
    """PEFT stores dropout as float or nn.Dropout depending on version."""
    if dropout is None:
        return 0.0
    if isinstance(dropout, (int, float)):
        return float(dropout)
    p = getattr(dropout, "p", None)
    return float(p) if p is not None else 0.0


def _is_peft_lora_linear(module: Any) -> bool:
    return (
        hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and hasattr(module, "base_layer")
        and hasattr(module, "scaling")
    )


def _is_qkv_projection(module_name: str) -> bool:
    return any(module_name.endswith(suffix) for suffix in _QKV_PROJECTION_SUFFIXES)


def apply_fused_lora_kernels(
    model: Any,
    *,
    max_rank: int = 64,
    low_vram: bool | None = None,
    skip_qkv: bool = False,
) -> dict[str, Any]:
    """
    Patch PEFT LoRA linear layers with fused CUDA low-rank delta kernels.

    Active on NVIDIA CUDA and WSL2 paths when rank <= max_rank.
    In low-VRAM mode, writes the LoRA delta in-place into the base output.
    """
    if low_vram is None:
        low_vram = kernel_low_vram_enabled()

    meta = {
        "fused_lora_enabled": False,
        "lora_patched": 0,
        "lora_skipped": 0,
        "kernel_low_vram": low_vram,
    }
    platform = detect_gpu()
    if not platform.uses_optimized_cuda_kernels or active_backend() != "cuda":
        return meta

    try:
        import torch.nn.functional as F
    except ImportError:
        return meta

    patched = 0
    skipped = 0

    for name, module in model.named_modules():
        if not _is_peft_lora_linear(module):
            continue
        if skip_qkv and _is_qkv_projection(name):
            continue
        if getattr(module, "use_dora", False):
            skipped += 1
            continue
        if hasattr(module, "_seiso_orig_forward"):
            continue

        def _fused_lora_forward(self, x, *args, **kwargs):
            if self.disable_adapters:
                return self.base_layer(x, *args, **kwargs)

            result = self.base_layer(x, *args, **kwargs)
            if not self.active_adapters:
                return result

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A:
                    continue
                lora_a = self.lora_A[active_adapter]
                lora_b = self.lora_B[active_adapter]
                dropout_p = _lora_dropout_p(self.lora_dropout[active_adapter])
                scaling = self.scaling[active_adapter]
                rank = lora_a.weight.size(0)

                x_mod = x
                if hasattr(self, "_cast_input_dtype"):
                    x_mod = self._cast_input_dtype(x_mod, lora_a.weight.dtype)
                if dropout_p > 0 and self.training:
                    x_mod = F.dropout(x_mod, p=dropout_p)

                if rank <= max_rank and x_mod.dim() >= 2:
                    flat_x = x_mod.reshape(-1, x_mod.shape[-1])
                    import torch

                    if low_vram and not torch.is_grad_enabled():
                        flat_out = result.reshape(-1, result.shape[-1])
                        fused_lora_delta(
                            flat_x,
                            lora_a.weight,
                            lora_b.weight,
                            base=flat_out,
                            scale=scaling,
                            inplace=True,
                        )
                    else:
                        delta = fused_lora_delta(
                            flat_x, lora_a.weight, lora_b.weight, scale=scaling
                        )
                        delta = delta.reshape(*x_mod.shape[:-1], delta.shape[-1])
                        result = result + delta.to(result.dtype)
                else:
                    result = result + lora_b(lora_a(x_mod)) * scaling

            return result

        _patch_forward(model, module, _fused_lora_forward)
        patched += 1

    meta["fused_lora_enabled"] = patched > 0
    meta["lora_patched"] = patched
    meta["lora_skipped"] = skipped
    if patched:
        target = "WSL2 CUDA" if platform.is_wsl2 else "CUDA"
        logger.info("Fused LoRA: %d layers patched (%s path)", patched, target)
    return meta


def _einsum_linear_batch(
    flat_x: Any,
    weights: tuple[Any, ...],
    biases: tuple[Any | None, ...],
) -> tuple[Any, ...]:
    """Batched linear via one einsum when shapes match."""
    import torch

    W = torch.stack(weights, dim=0)
    outs = torch.einsum("soi,bi->bso", W, flat_x)
    if all(b is not None for b in biases):
        outs = outs + torch.stack(biases, dim=0).unsqueeze(0)
    return outs.unbind(dim=1)


def _layer_bias(layer: Any) -> Any | None:
    return getattr(layer, "bias", None)


def _supports_einsum_batch(layers: tuple[Any, ...]) -> bool:
    """Quantized (bitsandbytes) base layers must use their own forward."""
    for layer in layers:
        cls = type(layer).__name__.lower()
        if any(token in cls for token in ("bnb", "bitsandbytes", "linear4bit", "linear8bit")):
            return False
        weight = getattr(layer, "weight", None)
        if weight is not None and hasattr(weight, "quant_state"):
            return False
    return True


def _bias_set_complete(biases: tuple[Any | None, ...]) -> bool:
    return not any(b is not None for b in biases) or all(b is not None for b in biases)


def _batched_base_qkv_forward(
    flat_x: Any, q_proj: Any, k_proj: Any, v_proj: Any
) -> tuple[Any, Any, Any]:
    """Batched base Q/K/V linear — full MHA or GQA (shared K/V shapes)."""
    layers = (q_proj.base_layer, k_proj.base_layer, v_proj.base_layer)
    weights = tuple(ly.weight for ly in layers)
    biases = tuple(_layer_bias(ly) for ly in layers)

    if not _bias_set_complete(biases) or not _supports_einsum_batch(layers):
        return (
            q_proj.base_layer(flat_x),
            k_proj.base_layer(flat_x),
            v_proj.base_layer(flat_x),
        )

    if weights[0].shape == weights[1].shape == weights[2].shape:
        return _einsum_linear_batch(flat_x, weights, biases)

    if weights[1].shape == weights[2].shape:
        out_q = q_proj.base_layer(flat_x)
        out_k, out_v = _einsum_linear_batch(
            flat_x,
            (weights[1], weights[2]),
            (biases[1], biases[2]),
        )
        return out_q, out_k, out_v

    return (
        q_proj.base_layer(flat_x),
        k_proj.base_layer(flat_x),
        v_proj.base_layer(flat_x),
    )


def _get_active_lora_weights(module: Any, adapter: str) -> tuple[Any, Any, float] | None:
    if adapter not in module.lora_A or adapter not in module.lora_B:
        return None
    return (
        module.lora_A[adapter].weight,
        module.lora_B[adapter].weight,
        float(module.scaling[adapter]),
    )


def _patch_fused_qkv_projections(
    model: Any,
    attn_module: Any,
    *,
    max_rank: int,
) -> bool:
    """Coordinator patches q/k/v PEFT layers to share one fused CUDA LoRA pass."""
    cache: dict[str, Any] = {"key": None, "outs": None}

    def _proj_slot(proj: Any) -> int:
        if proj is attn_module.q_proj:
            return 0
        if proj is attn_module.k_proj:
            return 1
        return 2

    def _apply_adapter_qkv_delta(flat_x, out_q, out_k, out_v, adapter: str) -> bool:
        weights = [
            _get_active_lora_weights(p, adapter)
            for p in (attn_module.q_proj, attn_module.k_proj, attn_module.v_proj)
        ]
        if not all(weights) or any(w[0].size(0) > max_rank for w in weights):
            return False
        fused_lora_qkv_delta(
            flat_x,
            out_q,
            out_k,
            out_v,
            weights[0][0],
            weights[0][1],
            weights[1][0],
            weights[1][1],
            weights[2][0],
            weights[2][1],
            scale_q=weights[0][2],
            scale_k=weights[1][2],
            scale_v=weights[2][2],
        )
        return True

    def _make_proj_forward(proj: Any):
        def _fallback_projection(self, x, *args, **kwargs):
            import torch.nn.functional as F

            result = self.base_layer(x, *args, **kwargs)
            for adapter in self.active_adapters:
                if adapter not in self.lora_A:
                    continue
                x_mod = x
                if hasattr(self, "_cast_input_dtype"):
                    x_mod = self._cast_input_dtype(x_mod, self.lora_A[adapter].weight.dtype)
                dropout_p = _lora_dropout_p(self.lora_dropout[adapter])
                if dropout_p > 0 and self.training:
                    x_mod = F.dropout(x_mod, p=dropout_p)
                flat_x = x_mod.reshape(-1, x_mod.shape[-1])
                w = _get_active_lora_weights(self, adapter)
                if w and w[0].size(0) <= max_rank:
                    delta = fused_lora_delta(flat_x, w[0], w[1], scale=w[2])
                    result = result + delta.reshape(*x_mod.shape[:-1], delta.shape[-1]).to(
                        result.dtype
                    )
                else:
                    result = (
                        result
                        + self.lora_B[adapter](self.lora_A[adapter](x_mod)) * self.scaling[adapter]
                    )
            return result

        def _forward(self, x, *args, **kwargs):
            import torch.nn.functional as F

            if self.disable_adapters or not self.active_adapters:
                return self.base_layer(x, *args, **kwargs)

            if _proj_slot(self) == 0:
                cache["key"] = None
                cache["outs"] = None
                cache["x_mod"] = None

            adapters = tuple(self.active_adapters)
            if len(adapters) != 1:
                return _fallback_projection(self, x, *args, **kwargs)

            adapter = adapters[0]
            # Apply dropout once on q_proj (slot 0) and reuse for k/v so the
            # shared QKV cache key is stable. Per-proj dropout allocates distinct
            # tensors and breaks the data_ptr cache under default lora_dropout.
            if _proj_slot(self) == 0:
                x_mod = x
                if hasattr(self, "_cast_input_dtype"):
                    x_mod = self._cast_input_dtype(
                        x_mod, self.lora_A[adapter].weight.dtype
                    )
                dropout_p = _lora_dropout_p(self.lora_dropout[adapter])
                if dropout_p > 0 and self.training:
                    x_mod = F.dropout(x_mod, p=dropout_p)
                cache["x_mod"] = x_mod
            else:
                x_mod = cache.get("x_mod")
                if x_mod is None:
                    return _fallback_projection(self, x, *args, **kwargs)

            flat_x = x_mod.reshape(-1, x_mod.shape[-1])
            cache_key = (
                flat_x.data_ptr(),
                tuple(flat_x.shape),
                flat_x._version,
                adapters,
            )
            if cache["key"] != cache_key:
                out_q, out_k, out_v = _batched_base_qkv_forward(
                    flat_x,
                    attn_module.q_proj,
                    attn_module.k_proj,
                    attn_module.v_proj,
                )
                try:
                    if not _apply_adapter_qkv_delta(flat_x, out_q, out_k, out_v, adapter):
                        return _fallback_projection(self, x, *args, **kwargs)
                    cache["outs"] = (out_q, out_k, out_v)
                    cache["key"] = cache_key
                except (RuntimeError, ImportError):
                    return _fallback_projection(self, x, *args, **kwargs)

            outs = cache["outs"]
            if not isinstance(outs, tuple):
                return self.base_layer(x, *args, **kwargs)
            slot = _proj_slot(self)
            projected = outs[slot]  # pylint: disable=unsubscriptable-object
            return projected.reshape(*x.shape[:-1], projected.shape[-1]).to(x.dtype)

        return _forward

    for proj in (attn_module.q_proj, attn_module.k_proj, attn_module.v_proj):
        if hasattr(proj, "_seiso_orig_forward"):
            return False
        _patch_forward(model, proj, _make_proj_forward(proj))
    return True


def apply_fused_lora_qkv_kernels(
    model: Any, *, max_rank: int = 64, low_vram: bool | None = None
) -> dict[str, Any]:
    """
    Fuse LoRA Q/K/V delta computation per attention layer (single input read).

    Active on NVIDIA CUDA tensors when q/k/v PEFT layers share rank <= max_rank.
    If native CUDA extensions are unavailable, dispatch falls back to Torch/cuBLAS.
    """
    if low_vram is None:
        low_vram = kernel_low_vram_enabled()

    meta = {
        "fused_lora_qkv_enabled": False,
        "lora_qkv_patched": 0,
        "kernel_low_vram": low_vram,
    }
    if not detect_gpu().uses_optimized_cuda_kernels or active_backend() not in {
        "cuda",
        "triton",
    }:
        return meta

    patched = 0
    for _name, module in model.named_modules():
        if type(module).__name__ not in _ATTENTION_CLASSES:
            continue
        if not all(hasattr(module, p) for p in ("q_proj", "k_proj", "v_proj")):
            continue
        if not all(_is_peft_lora_linear(m) for m in (module.q_proj, module.k_proj, module.v_proj)):
            continue
        if _patch_fused_qkv_projections(model, module, max_rank=max_rank):
            patched += 1

    meta["fused_lora_qkv_enabled"] = patched > 0
    meta["lora_qkv_patched"] = patched
    if patched:
        logger.info("Fused LoRA QKV: %d attention layers patched", patched)
    return meta


def _patch_post_attention_residual_norm(model: Any, decoder: Any) -> bool:
    """Upgrade post_attention_layernorm to fuse residual+RMSNorm when residual is cached."""
    norm = decoder.post_attention_layernorm
    if hasattr(norm, "_seiso_residual_norm_forward"):
        return False

    fallback = norm.forward

    def _residual_norm_forward(self_norm, hidden_states, _parent=decoder, _fallback=fallback):
        if _use_fused_cuda_kernels(hidden_states):
            residual = getattr(_parent, "_seiso_residual", None)
            if (
                residual is not None
                and residual is not hidden_states
                and getattr(residual, "data_ptr", lambda: None)()
                != getattr(hidden_states, "data_ptr", lambda: None)()
            ):
                out = fused_rms_norm(
                    hidden_states,
                    self_norm.weight,
                    eps=getattr(self_norm, "variance_epsilon", getattr(self_norm, "eps", 1e-6)),
                    residual=residual,
                )
                _parent._seiso_residual = None
                return out
            # Decoder chose the fused branch; never drop residual on fallback.
            if residual is not None and residual is not hidden_states:
                _parent._seiso_residual = None
                return _fallback(residual + hidden_states)
            return _fallback(hidden_states)
        if hasattr(self_norm, "_seiso_orig_forward"):
            return self_norm._seiso_orig_forward(hidden_states)
        return _fallback(hidden_states)

    norm._seiso_residual_norm_forward = _residual_norm_forward
    if not hasattr(norm, "_seiso_orig_forward"):
        norm._seiso_orig_forward = fallback
    try:
        norm.forward = types.MethodType(_residual_norm_forward, norm)
        register_patch(model, norm)
    except Exception:
        norm.forward = norm._seiso_orig_forward
        delattr(norm, "_seiso_orig_forward")
        raise
    return True


def _patch_fused_residual_decoder_forward(model: Any, decoder: Any) -> bool:
    """Rewrite standard pre-norm decoder forwards to expose residual for fused post-attn norm."""
    if hasattr(decoder, "_seiso_residual_decoder_forward"):
        return False

    orig = decoder.forward
    attn_dropout = getattr(decoder, "resid_attn_dropout", None)
    mlp_dropout = getattr(decoder, "resid_mlp_dropout", None)

    def _decoder_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out

        attn_hidden = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        attn_for_norm = attn_dropout(attn_hidden) if attn_dropout is not None else attn_hidden
        post_attn_skip = residual + attn_for_norm

        if _use_fused_cuda_kernels(attn_for_norm):
            self._seiso_residual = residual
            hidden_states = self.post_attention_layernorm(attn_for_norm)
            self._seiso_residual = None
        else:
            hidden_states = self.post_attention_layernorm(post_attn_skip)

        hidden_states = self.mlp(hidden_states)
        if mlp_dropout is not None:
            hidden_states = post_attn_skip + mlp_dropout(hidden_states)
        else:
            hidden_states = post_attn_skip + hidden_states

        return hidden_states

    decoder._seiso_residual_decoder_forward = _decoder_forward
    decoder._seiso_orig_forward = orig
    try:
        decoder.forward = types.MethodType(_decoder_forward, decoder)
        register_patch(model, decoder)
    except Exception:
        decoder.forward = decoder._seiso_orig_forward
        delattr(decoder, "_seiso_orig_forward")
        raise
    return True


def apply_fused_residual_norm_kernels(model: Any) -> dict[str, Any]:
    """Patch decoder layers to fuse residual+RMSNorm on post-attention norms."""
    meta = {"fused_residual_norm_patched": 0, "fused_residual_decoder_patched": 0}
    if active_backend() not in {"cuda", "triton"}:
        return meta

    norm_patched = 0
    decoder_patched = 0
    for _name, module in model.named_modules():
        cls = type(module).__name__
        if cls not in _DECODER_LAYER_CLASSES:
            continue
        if not all(
            hasattr(module, a)
            for a in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp")
        ):
            continue

        if cls not in _FUSED_RESIDUAL_DECODER_CLASSES:
            continue

        if _patch_post_attention_residual_norm(model, module):
            norm_patched += 1

        if _patch_fused_residual_decoder_forward(model, module):
            decoder_patched += 1

    meta["fused_residual_norm_patched"] = norm_patched
    meta["fused_residual_decoder_patched"] = decoder_patched
    if norm_patched or decoder_patched:
        logger.info(
            "Fused residual+RMSNorm: %d post-attn norms, %d decoder forwards",
            norm_patched,
            decoder_patched,
        )
    return meta
