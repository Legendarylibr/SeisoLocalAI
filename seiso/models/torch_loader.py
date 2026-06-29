"""PyTorch/CUDA/MPS model loader with optional quantization."""

from __future__ import annotations

import logging
import platform
from typing import Any

from seiso.models.loader import Backend, LoadOptions

logger = logging.getLogger(__name__)


def _resolve_dtype(options: LoadOptions) -> Any:
    if not options.dtype:
        return None
    import torch

    return getattr(torch, options.dtype, None)


_AUTO_DEVICE_MAPS = {"auto", "balanced", "balanced_low_0", "sequential"}
_DISABLED_DEVICE_MAPS = {"", "none", "false", "off", "disabled", "cpu"}


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _mps_available() -> bool:
    import torch

    return bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def _resolve_device_map(
    backend: Backend,
    device: str | None = None,
    *,
    for_training: bool = False,
    requested: str | dict[str, str] | None = "auto",
) -> str | dict[str, str] | None:
    if for_training:
        from seiso.memory.protection import resolve_training_device_map

        return resolve_training_device_map(device)

    if isinstance(requested, dict):
        return requested

    request = "auto" if requested is None else str(requested).strip().lower()
    if request in _DISABLED_DEVICE_MAPS:
        return None

    if request == "cuda":
        return {"": "cuda"} if _cuda_available() else None
    if request == "mps":
        return {"": "mps"} if _mps_available() else None

    if request in _AUTO_DEVICE_MAPS:
        if device == "cuda" or (backend == Backend.TORCH and _cuda_available()):
            return request
        if request == "auto" and (device == "mps" or _mps_available()):
            return {"": "mps"}
        return None

    return request


def _device_map_accepts_max_memory(device_map: str | dict[str, str] | None) -> bool:
    return isinstance(device_map, str) and device_map in _AUTO_DEVICE_MAPS


def _looks_like_quantization_load_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "bitsandbytes",
            "bnb",
            "quantization_config",
            "load_in_4bit",
            "load_in_8bit",
            "4-bit",
            "8-bit",
            "quantized",
            "cuda is required",
        )
    )


def load_torch(
    options: LoadOptions,
    backend: Backend = Backend.TORCH,
    *,
    device: str | None = None,
    for_training: bool = False,
) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from seiso.models.hf_env import _read_hub_token

    hub_token = _read_hub_token()
    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": options.trust_remote_code,
        "revision": options.revision or "main",
    }
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": options.trust_remote_code,
        "revision": options.revision or "main",
    }
    if hub_token:
        tokenizer_kwargs["token"] = hub_token
        model_kwargs["token"] = hub_token

    from seiso.models.hub_quant import native_quant_method_from_config

    pre_config = AutoConfig.from_pretrained(options.model_id, **tokenizer_kwargs)
    pre_quant_method = native_quant_method_from_config(pre_config) or ""
    native_hub_quant = bool(pre_quant_method)

    device_map = _resolve_device_map(
        backend,
        device,
        for_training=for_training,
        requested=options.device_map,
    )
    if device_map is not None:
        model_kwargs["device_map"] = device_map
        if _device_map_accepts_max_memory(device_map):
            from seiso.memory.protection import build_hf_max_memory

            max_memory = build_hf_max_memory()
            if max_memory:
                model_kwargs["max_memory"] = max_memory

    dtype = _resolve_dtype(options)
    if dtype is None and device != "mps" and not native_hub_quant:
        try:
            import torch

            if torch.cuda.is_available():
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
        except ImportError:
            dtype = None
    if dtype is not None and not native_hub_quant:
        model_kwargs["torch_dtype"] = dtype

    model_kwargs.setdefault("low_cpu_mem_usage", True)

    if native_hub_quant:
        model_kwargs["attn_implementation"] = "eager"
        logger.info(
            "Native %s weights — keeping hub dtype, using eager attention",
            pre_quant_method,
        )
    elif options.use_flash_attention and device != "mps":
        from seiso.kernels.attention import resolve_attention_implementation

        attn_impl = resolve_attention_implementation(prefer_fa3=True)
        model_kwargs["attn_implementation"] = attn_impl
        logger.info("Using %s attention", attn_impl.replace("_", " "))

    use_4bit = options.load_in_4bit
    use_8bit = options.load_in_8bit

    if native_hub_quant:
        logger.info(
            "Model ships with native %s weights — skipping bitsandbytes QLoRA quant",
            pre_quant_method,
        )
        use_4bit = False
        use_8bit = False

    if use_4bit or use_8bit:
        # bitsandbytes (required for 4-bit/8-bit quantization) is only available
        # on Linux/Windows with a CUDA or ROCm GPU. On macOS (no bnb) or a
        # CPU-only box, silently fall back to unquantized load instead of
        # crashing at BitsAndBytesConfig or model.from_pretrained.
        bnb_unavailable = platform.system() == "Darwin"
        if not bnb_unavailable:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                bnb_unavailable = True
            else:
                try:
                    import torch

                    if not torch.cuda.is_available():
                        bnb_unavailable = True
                except ImportError:
                    bnb_unavailable = True
        if bnb_unavailable:
            logger.warning(
                "bitsandbytes quantization unavailable on this platform — "
                "loading without quantization"
            )
            use_4bit = False
            use_8bit = False

    quantization_requested = False
    if use_4bit:
        from transformers import BitsAndBytesConfig

        quantization_requested = True
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_kwargs.get("torch_dtype"),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        from transformers import BitsAndBytesConfig

        quantization_requested = True
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        options.model_id, **tokenizer_kwargs
    )  # nosec B615: revision pinned in tokenizer_kwargs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            options.model_id, **model_kwargs
        )  # nosec B615: revision pinned in model_kwargs
    except Exception as exc:
        if not quantization_requested or not _looks_like_quantization_load_error(exc):
            raise
        retry_kwargs = dict(model_kwargs)
        retry_kwargs.pop("quantization_config", None)
        logger.warning(
            "Quantized torch load failed (%s) - retrying without bitsandbytes "
            "quantization",
            exc,
        )
        model = AutoModelForCausalLM.from_pretrained(
            options.model_id, **retry_kwargs
        )  # nosec B615: revision pinned in model_kwargs

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
