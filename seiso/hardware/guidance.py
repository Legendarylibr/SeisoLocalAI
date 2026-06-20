"""Hardware-aware onboarding guidance steps."""

from __future__ import annotations

from dataclasses import dataclass

from seiso.models.loader import Backend


@dataclass
class GuideStep:
    title: str
    detail: str
    path: str
    priority: int = 0


def build_guidance(goal: str, *, backend: Backend, gpus: list[dict], ram_gb: float) -> list[GuideStep]:
    """Hardware-aware next steps — no cloud, no data collection."""
    vram_total = max((g.get("vram_total_mb") or 0) for g in gpus) if gpus else 0
    steps: list[GuideStep] = []

    if goal == "chat":
        if vram_total >= 12000 or backend == Backend.MLX:
            steps.append(GuideStep("Download a 7–14B model", "Your hardware can run strong chat models locally.", "/hub", 3))
            steps.append(GuideStep("Open Chat", "Start a new conversation with encrypted session memory.", "/chat", 2))
        elif vram_total >= 6000:
            steps.append(GuideStep("Get a 3–7B GGUF", "Quantized models fit comfortably in your VRAM.", "/hub", 3))
            steps.append(GuideStep("Open Chat", "Pick a model and start chatting.", "/chat", 2))
        elif gpus:
            steps.append(GuideStep("Try a 1–3B model", "Limited VRAM — use Q4_K_M quantization.", "/hub", 3))
        else:
            steps.append(GuideStep("Try a small GGUF", "No discrete GPU — use Q4 models with llama.cpp.", "/hub", 3))
        steps.append(GuideStep("Monitor load", "Watch GPU/CPU in the live monitor (stays on this machine).", "/", 1))

    elif goal == "train":
        if vram_total >= 24000:
            steps.append(GuideStep("Fine-tune 7B+", "Enough VRAM for LoRA on 7–14B models.", "/train", 3))
        elif vram_total >= 12000:
            steps.append(GuideStep("LoRA on 3–7B", "Use 4-bit loading and small batch sizes.", "/train", 3))
        else:
            steps.append(GuideStep("Consider QLoRA or cloud export", "Training needs more VRAM — try compress or export flows.", "/compress", 2))
        if ram_gb < 16:
            steps.append(GuideStep("Close other apps", f"System RAM is {ram_gb} GB — training benefits from 16 GB+.", "/", 1))

    elif goal == "compress":
        steps.append(GuideStep("Compress a checkpoint", "Prune, quantize, and export smaller models.", "/compress", 3))
        if vram_total < 8000:
            steps.append(GuideStep("Start small", "Use smoke presets on CPU if GPU memory is tight.", "/compress", 2))

    elif goal in ("inference", "code"):
        steps.append(GuideStep("Browse models", "Download a GGUF from the Hub — newest models listed first.", "/hub", 3))
        steps.append(GuideStep("Open Chat", "Run local inference via llama.cpp, Ollama, or MLX.", "/chat", 3))
        if vram_total < 6000 and not gpus:
            steps.append(GuideStep("Download a Q4 GGUF", "llama.cpp runs small quantized models efficiently on CPU.", "/hub", 2))

    else:
        steps.append(GuideStep("Browse the catalog", "Newest models are listed first.", "/hub", 2))
        steps.append(GuideStep("Open Chat", "Encrypted memory lasts until you sign out.", "/chat", 2))

    steps.sort(key=lambda s: -s.priority)
    return steps
