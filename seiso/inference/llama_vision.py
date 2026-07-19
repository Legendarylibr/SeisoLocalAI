"""llama.cpp multimodal (mmproj) helpers for vision-capable GGUF chat models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_QUANT_HINTS = (
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q4_K_M",
    "Q4_0",
    "IQ4_XS",
    "F16",
    "F32",
    "BF16",
)
_VISION_NAME_MARKERS = (
    "vision",
    "-vl",
    "vl-",
    "llava",
    "pixtral",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "minicpm-v",
    "smolvlm",
    "moondream",
)
_GEMMA_VISION_MARKERS = (
    "gemma-vision",
    "gemma3-vision",
    "gemma-3-vision",
    "gemma4-vision",
    "gemma-4-vision",
)


def _name_suggests_vision(name: str) -> bool:
    hay = name.lower()
    if any(marker in hay for marker in _VISION_NAME_MARKERS):
        return True
    return any(marker in hay for marker in _GEMMA_VISION_MARKERS)


def model_suggests_vision(model_path: str | Path) -> bool:
    """True when the chat model itself looks vision-capable (not just a sibling mmproj)."""
    name = Path(model_path).name.lower()
    if _name_suggests_vision(name):
        return True
    from seiso.inference.backends import gguf_architecture

    arch = (gguf_architecture(str(model_path)) or "").lower()
    return any(token in arch for token in ("vision", "clip", "mmproj", "vl"))


def repo_likely_needs_mmproj(
    catalog_repo_id: str,
    *,
    gguf_filename: str | None = None,
    tags: tuple[str, ...] | list[str] | None = None,
    task: str | None = None,
) -> bool:
    """True when a Hub download should also fetch a colocated mmproj file."""
    tag_set = {str(tag).lower() for tag in (tags or ())}
    if "vision" in tag_set or "multimodal" in tag_set:
        return True
    if task and str(task).lower() == "vision":
        return True
    if gguf_filename and _name_suggests_vision(gguf_filename):
        return True
    return _name_suggests_vision(catalog_repo_id)


def _quant_hints_from_name(name: str) -> list[str]:
    upper = name.upper()
    return [hint for hint in _QUANT_HINTS if hint in upper]


def _mmproj_candidates(parent: Path) -> list[Path]:
    return sorted(
        (
            item
            for item in parent.glob("*.gguf")
            if "mmproj" in item.name.lower()
        ),
        key=lambda item: item.name,
    )


def resolve_mmproj_path(model_path: str | Path) -> str | None:
    """Return the best colocated mmproj GGUF for a chat model, if present."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        return None
    candidates = _mmproj_candidates(path.parent)
    if not candidates:
        return None
    if len(candidates) == 1:
        return str(candidates[0].resolve())

    model_hints = _quant_hints_from_name(path.name)
    for hint in model_hints:
        matched = [item for item in candidates if hint in item.name.upper()]
        if matched:
            return str(matched[0].resolve())
    return str(candidates[0].resolve())


def _vision_handler_specs(model_path: str) -> list[str]:
    """Ordered llama-cpp-python chat handler class names to try."""
    from seiso.inference.backends import gguf_architecture

    arch = (gguf_architecture(model_path) or "").lower()
    name = Path(model_path).name.lower()
    specs: list[str] = []

    if "qwen" in name and ("vl" in name or "vision" in arch):
        specs.append("Qwen25VLChatHandler")
    if ("gemma" in name or "gemma" in arch) and (
        _name_suggests_vision(name)
        or any(token in arch for token in ("vision", "clip", "mmproj", "vl"))
    ):
        specs.extend(["Gemma4ChatHandler", "Gemma3ChatHandler"])
    if "llava" in name or "llava" in arch:
        specs.extend(["Llava16ChatHandler", "Llava15ChatHandler"])
    if "moondream" in name:
        specs.append("MoondreamChatHandler")
    if "minicpm" in name and "v" in name:
        specs.append("MiniCPMv26ChatHandler")

    specs.append("MTMDChatHandler")
    seen: set[str] = set()
    ordered: list[str] = []
    for item in specs:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def build_llama_vision_chat_handler(model_path: str, mmproj_path: str) -> Any | None:
    """Build a llama.cpp vision chat handler when mmproj is available."""
    if not mmproj_path or not Path(mmproj_path).is_file():
        return None

    try:
        import llama_cpp.llama_chat_format as chat_format
    except ImportError:
        logger.debug("llama_cpp.llama_chat_format unavailable — skipping vision handler")
        return None

    last_exc: Exception | None = None
    warned = False
    for class_name in _vision_handler_specs(model_path):
        cls = getattr(chat_format, class_name, None)
        if cls is None:
            continue
        try:
            return cls(clip_model_path=mmproj_path, verbose=False)
        except ImportError:
            continue
        except Exception as exc:
            last_exc = exc
            if not warned:
                logger.warning(
                    "Vision handler %s failed for %s: %s",
                    class_name,
                    Path(model_path).name,
                    exc,
                )
                warned = True
            logger.debug(
                "Vision handler %s failed for %s: %s", class_name, model_path, exc
            )

    if last_exc is not None:
        logger.warning(
            "Could not initialize vision handler for %s (mmproj=%s): %s",
            Path(model_path).name,
            Path(mmproj_path).name,
            last_exc,
        )
    return None


def apply_llama_vision_load_kwargs(
    load_kwargs: dict[str, Any], model_path: str
) -> dict[str, Any]:
    """Attach a vision chat handler to llama.cpp load kwargs when appropriate."""
    mmproj = resolve_mmproj_path(model_path)
    if not mmproj:
        return load_kwargs
    # Text-only models in a shared download dir must not pick up a sibling mmproj.
    if not model_suggests_vision(model_path):
        return load_kwargs
    handler = build_llama_vision_chat_handler(model_path, mmproj)
    if handler is None:
        return load_kwargs
    out = dict(load_kwargs)
    out["chat_handler"] = handler
    return out
