"""Local-only hardware detection and live metrics — no telemetry, no external calls."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from seiso.models.catalog import _parse_param_size
from seiso.models.loader import Backend, detect_backend

# Strip serial numbers / host-specific identifiers from hardware strings.
_SERIAL_RE = re.compile(r"\b(serial|s/n|uuid)[:\s#-]*[\w-]+", re.I)
_HOST_RE = re.compile(r"@[\w.-]+")
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b", re.I)
_ACTIVE_MOE_RE = re.compile(r"a(\d+(?:\.\d+)?)b", re.I)

INFERENCE_BACKEND_LABELS: dict[str, str] = {
    "llamacpp": "llama.cpp",
    "ollama": "Ollama",
    "mlx": "MLX",
    "torch": "PyTorch",
    "auto": "Auto",
}

FIT_RANK = {"ideal": 4, "good": 3, "tight": 2, "unlikely": 1}

_PROFILE_TTL_S = 30.0
_METRICS_TTL_S = 1.5
_profile_cache: dict[str, Any] | None = None
_profile_cache_ts: float = 0.0
_metrics_cache: dict[str, Any] | None = None
_metrics_cache_ts: float = 0.0
_cpu_percent_primed = False


class HardwareTier(StrEnum):
    CPU_ONLY = "cpu_only"
    EDGE = "edge"
    MODEST = "modest"
    CAPABLE = "capable"
    WORKSTATION = "workstation"
    APPLE_UNIFIED = "apple_unified"


TIER_LABELS: dict[HardwareTier, str] = {
    HardwareTier.CPU_ONLY: "CPU only",
    HardwareTier.EDGE: "Edge GPU",
    HardwareTier.MODEST: "Modest GPU",
    HardwareTier.CAPABLE: "Capable GPU",
    HardwareTier.WORKSTATION: "Workstation GPU",
    HardwareTier.APPLE_UNIFIED: "Apple unified memory",
}


def _sanitize_label(raw: str, *, max_len: int = 64) -> str:
    """Return a generic hardware label safe to show in the UI."""
    text = _SERIAL_RE.sub("", raw)
    text = _HOST_RE.sub("", text)
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text or "Unknown"


def _ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass
    try:
        page_size = __import__("os").sysconf("SC_PAGE_SIZE")
        phys_pages = __import__("os").sysconf("SC_PHYS_PAGES")
        return round((page_size * phys_pages) / (1024**3), 1)
    except (AttributeError, OSError, ValueError):
        return 0.0


def _cpu_brand() -> str:
    try:
        import cpuinfo  # type: ignore

        brand = cpuinfo.get_cpu_info().get("brand_raw", "")
        if brand:
            return _sanitize_label(brand)
    except ImportError:
        pass
    proc = platform.processor() or platform.machine()
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "Apple Silicon"
    return _sanitize_label(proc)


def _cpu_cores() -> int:
    return __import__("os").cpu_count() or 1


def _torch_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return gpus
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            name = _sanitize_label(props.name)
            total_mb = int(props.total_memory / (1024**2))
            used_mb: int | None = None
            util: float | None = None
            temp: float | None = None
            try:
                free, total = torch.cuda.mem_get_info(i)
                used_mb = int((total - free) / (1024**2))
            except Exception:
                pass
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "vram_total_mb": total_mb,
                    "vram_used_mb": used_mb,
                    "utilization_pct": util,
                    "temperature_c": temp,
                }
            )
    except ImportError:
        pass
    return gpus


def _nvidia_smi_metrics() -> dict[int, dict[str, float]]:
    """Parse nvidia-smi locally — never leaves the machine."""
    out: dict[int, dict[str, float]] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            return out
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            idx = int(parts[0])
            out[idx] = {
                "utilization_pct": float(parts[1]) if parts[1] not in ("[N/A]", "N/A") else 0.0,
                "vram_used_mb": float(parts[2]),
                "vram_total_mb": float(parts[3]),
                "temperature_c": float(parts[4]) if parts[4] not in ("[N/A]", "N/A") else 0.0,
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return out


def _mlx_apple_gpu() -> list[dict[str, Any]]:
    """Best-effort Apple GPU label when MLX is available."""
    try:
        import mlx.core as mx  # noqa: F401

        return [
            {
                "index": 0,
                "name": "Apple GPU (MLX)",
                "vram_total_mb": None,
                "vram_used_mb": None,
                "utilization_pct": None,
                "temperature_c": None,
            }
        ]
    except ImportError:
        return []


def detect_gpus() -> list[dict[str, Any]]:
    gpus = _torch_gpus()
    if not gpus:
        gpus = _mlx_apple_gpu()
    smi = _nvidia_smi_metrics()
    for gpu in gpus:
        idx = gpu.get("index", 0)
        if idx in smi:
            gpu.update({k: v for k, v in smi[idx].items() if v is not None})
    return gpus


def live_metrics() -> dict[str, Any]:
    """Snapshot of CPU/RAM/GPU — aggregated locally, never exported."""
    global _metrics_cache, _metrics_cache_ts, _cpu_percent_primed

    now = time.time()
    if _metrics_cache is not None and now - _metrics_cache_ts < _METRICS_TTL_S:
        return _metrics_cache

    cpu_util: float | None = None
    cpu_temp: float | None = None
    ram_used_pct = 0.0

    try:
        import psutil  # type: ignore

        # First call primes the counter; subsequent calls use interval=None (non-blocking).
        cpu_util = round(
            psutil.cpu_percent(interval=None if _cpu_percent_primed else 0.05),
            1,
        )
        _cpu_percent_primed = True
        ram_used_pct = round(psutil.virtual_memory().percent, 1)
        sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
        temps = sensors_temperatures() if sensors_temperatures else {}
        for key in ("coretemp", "cpu_thermal", "TC0P", "TH0x"):
            if key in temps and temps[key]:
                cpu_temp = round(temps[key][0].current, 1)
                break
    except (ImportError, AttributeError):
        pass

    gpus = detect_gpus()

    result = {
        "cpu_util_pct": cpu_util,
        "cpu_temp_c": cpu_temp,
        "ram_used_pct": ram_used_pct,
        "gpus": gpus,
        "local_only": True,
        "ts": now,
    }
    _metrics_cache = result
    _metrics_cache_ts = now
    return result


def _vram_headroom_mb(gpus: list[dict[str, Any]]) -> int:
    if not gpus:
        return 0
    best = 0
    for g in gpus:
        total = g.get("vram_total_mb") or 0
        used = g.get("vram_used_mb") or 0
        best = max(best, int(total - used))
    return best


@dataclass
class GuideStep:
    title: str
    detail: str
    path: str
    priority: int = 0


def build_guidance(goal: str, *, backend: Backend, gpus: list[dict[str, Any]], ram_gb: float) -> list[GuideStep]:
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


def hardware_profile(*, force_refresh: bool = False) -> dict[str, Any]:
    global _profile_cache, _profile_cache_ts

    now = time.time()
    if not force_refresh and _profile_cache is not None and now - _profile_cache_ts < _PROFILE_TTL_S:
        return _profile_cache

    backend = detect_backend()
    gpus = detect_gpus()
    ram = _ram_gb()
    disk_free = shutil.disk_usage("/").free // (1024**3)

    profile = {
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "backend": backend.value,
        "cpu_cores": _cpu_cores(),
        "cpu_brand": _cpu_brand(),
        "ram_gb": ram,
        "disk_free_gb": disk_free,
        "gpus": gpus,
        "local_only": True,
        "privacy": "Metrics are read locally and never sent off this machine.",
    }
    _profile_cache = enrich_profile(profile)
    _profile_cache_ts = now
    return _profile_cache


# ── Hardware-aware recommendations (local heuristics, no cloud) ──


def _active_params_b(params: str, tags: tuple[str, ...] | list[str], repo_id: str = "") -> float:
    """Effective parameter count for VRAM estimates (MoE / active experts)."""
    text = f"{params} {repo_id}".lower()
    moe_match = _ACTIVE_MOE_RE.search(text)
    if moe_match:
        return float(moe_match.group(1))
    raw = _parse_param_size(params)
    if "moe" in tags:
        return max(raw * 0.2, 1.0)
    return raw


def _quant_bytes_per_param_b(quant: str) -> float:
    quant_u = quant.upper()
    if "Q8" in quant_u or "F16" in quant_u or "BF16" in quant_u:
        return 1.1
    if "Q5" in quant_u:
        return 0.75
    if "Q4" in quant_u or "IQ4" in quant_u:
        return 0.55
    return 0.65


def estimate_chat_vram_gb(params: str, *, quant: str = "Q4_K_M", tags: tuple[str, ...] | list[str] = (), repo_id: str = "") -> float:
    """Rough GGUF chat VRAM — conservative, for fit labels only."""
    params_b = _active_params_b(params, tags, repo_id)
    return round(params_b * _quant_bytes_per_param_b(quant) + 1.2, 2)


def estimate_gguf_download_bytes(
    params: str,
    *,
    quant: str = "Q4_K_M",
    tags: tuple[str, ...] | list[str] = (),
    repo_id: str = "",
) -> int:
    """Estimate on-disk GGUF size from active params and quant (fallback when Hub metadata unavailable)."""
    params_b = _active_params_b(params, tags, repo_id)
    gb = params_b * _quant_bytes_per_param_b(quant) + 0.4
    return int(max(gb, 0.25) * 1024**3)


def classify_tier(profile: dict[str, Any]) -> HardwareTier:
    raw_backend = profile.get("backend", "cpu")
    try:
        backend = Backend(raw_backend)
    except ValueError:
        backend = Backend.TORCH if raw_backend in ("cuda", "rocm") else Backend.CPU
    gpus = profile.get("gpus") or []
    vram_total = max((g.get("vram_total_mb") or 0) for g in gpus) if gpus else 0

    if backend == Backend.MLX and not vram_total:
        return HardwareTier.APPLE_UNIFIED
    if not gpus or vram_total <= 0:
        return HardwareTier.CPU_ONLY
    if vram_total >= 24000:
        return HardwareTier.WORKSTATION
    if vram_total >= 12000:
        return HardwareTier.CAPABLE
    if vram_total >= 6000:
        return HardwareTier.MODEST
    return HardwareTier.EDGE


def effective_budget_mb(profile: dict[str, Any]) -> int:
    """Memory budget for local inference — derived on-device only."""
    tier = classify_tier(profile)
    gpus = profile.get("gpus") or []
    ram = float(profile.get("ram_gb") or 0)
    vram_total = max((g.get("vram_total_mb") or 0) for g in gpus) if gpus else 0

    if tier == HardwareTier.APPLE_UNIFIED:
        return int(ram * 1024 * 0.55)
    if tier == HardwareTier.CPU_ONLY:
        return int(min(ram * 1024 * 0.35, 8192))
    return int(vram_total or min(ram * 1024 * 0.4, 8192))


def vram_headroom_mb(profile: dict[str, Any]) -> int:
    """Free memory headroom for fit checks."""
    gpus = profile.get("gpus") or []
    if gpus:
        best = _vram_headroom_mb(gpus)
        if best > 0:
            return best
    tier = classify_tier(profile)
    ram = float(profile.get("ram_gb") or 0)
    if tier in (HardwareTier.APPLE_UNIFIED, HardwareTier.CPU_ONLY):
        try:
            import psutil  # type: ignore

            avail = psutil.virtual_memory().available / (1024**2)
            return int(min(avail * 0.65, effective_budget_mb(profile)))
        except ImportError:
            return int(ram * 1024 * 0.4)
    return effective_budget_mb(profile)


def assess_hardware_fit(
    est_vram_gb: float,
    profile: dict[str, Any],
    *,
    mode: str = "chat",
) -> dict[str, Any]:
    """Return fit label + short note — never leaves the machine."""
    headroom_mb = vram_headroom_mb(profile)
    est_mb = int(est_vram_gb * 1024)
    tier = classify_tier(profile)

    if mode == "train":
        est_mb = int(est_mb * 2.2)

    ratio = est_mb / headroom_mb if headroom_mb > 0 else 99.0

    if ratio <= 0.65:
        fit, label = "ideal", "Ideal fit"
    elif ratio <= 0.95:
        fit, label = "good", "Good fit"
    elif ratio <= 1.15:
        fit, label = "tight", "Tight fit"
    else:
        fit, label = "unlikely", "May not fit"

    if tier == HardwareTier.CPU_ONLY and est_mb > 4096:
        fit, label = "unlikely", "CPU — try ≤3B Q4"

    if tier == HardwareTier.APPLE_UNIFIED and headroom_mb < 12288 and est_mb > 5120:
        fit, label = "tight", "Tight — use Q4 GGUF + llama.cpp"
    if tier == HardwareTier.APPLE_UNIFIED and headroom_mb < 8192 and est_mb > 4096:
        fit, label = "unlikely", "Low memory — try ≤3B Q4"

    headroom_gb = round(headroom_mb / 1024, 1)
    note = f"~{est_vram_gb:.1f} GB est. · {headroom_gb} GB free on this machine"
    if fit == "unlikely" and tier != HardwareTier.CPU_ONLY:
        note = f"Needs ~{est_vram_gb:.1f} GB — you have ~{headroom_gb} GB free"

    return {
        "hardware_fit": fit,
        "hardware_fit_label": label,
        "est_vram_mb": est_mb,
        "hardware_note": note,
        "hardware_fit_rank": FIT_RANK[fit],
    }


def _format_catalog_note(
    *,
    est_vram_gb: float,
    download_bytes: int,
    headroom_gb: float,
    fit: str,
    tier: HardwareTier,
) -> str:
    dl = f"Download ~{download_bytes / (1024**3):.1f} GB · " if download_bytes > 0 else ""
    runtime = f"Runtime ~{est_vram_gb:.1f} GB est. · "
    if fit == "unlikely" and tier != HardwareTier.CPU_ONLY:
        return f"{dl}{runtime}Needs ~{est_vram_gb:.1f} GB at runtime — you have ~{headroom_gb} GB free"
    return f"{dl}{runtime}{headroom_gb} GB free on this machine"


def assess_catalog_fit(model: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    est_gb = estimate_chat_vram_gb(
        model["params"],
        quant=model.get("quant", "Q4_K_M"),
        tags=model.get("tags", ()),
        repo_id=model.get("repo_id", ""),
    )
    mode = "train" if model.get("task") in ("base",) else "chat"
    return assess_hardware_fit(est_gb, profile, mode=mode)


def enrich_catalog_models(
    models: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    token: str | None = None,
    fetch_sizes: bool = True,
    diversify: bool = False,
) -> list[dict[str, Any]]:
    from forge.services.hf_hub import resolve_gguf_artifact
    from seiso.models.catalog import diversify_by_family, get_by_repo

    download_info: dict[str, dict[str, Any]] = {}
    download_errors: dict[str, str] = {}
    if models and fetch_sizes:
        # Anonymous HF API is aggressively rate-limited. Resolve only the visible
        # front of the catalog unless a token is configured.
        candidates = models if token else models[:16]
        workers = min(3 if token else 2, len(candidates))

        def fetch_info(repo_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
            try:
                return repo_id, resolve_gguf_artifact(
                    repo_id,
                    entry=get_by_repo(repo_id),
                    token=token,
                ), None
            except Exception as exc:
                return repo_id, None, str(exc)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_info, m["repo_id"]): m["repo_id"] for m in candidates}
            for future in as_completed(futures):
                repo_id, info, error = future.result()
                if info:
                    download_info[repo_id] = info
                elif error:
                    download_errors[repo_id] = error

    headroom_gb = round(vram_headroom_mb(profile) / 1024, 1)
    tier = classify_tier(profile)
    enriched: list[dict[str, Any]] = []
    for m in models:
        fit = assess_catalog_fit(m, profile)
        row = {**m, **fit}
        info = download_info.get(m["repo_id"])
        if info and info.get("size_bytes"):
            download_bytes = int(info["size_bytes"])
            actual_fit = assess_hardware_fit(
                round(download_bytes / (1024**3) + 0.8, 2),
                profile,
                mode="chat",
            )
            row.update(actual_fit)
            row["download_bytes"] = download_bytes
            row["download_bytes_estimated"] = False
            row["gguf_repo"] = info["gguf_repo"]
            row["gguf_file"] = info["filename"]
            row["download_available"] = True
        elif m["repo_id"] in download_errors:
            download_bytes = 0
            row["download_available"] = False
            row["download_error"] = download_errors[m["repo_id"]]
        elif m.get("task") != "embedding":
            download_bytes = estimate_gguf_download_bytes(
                m["params"],
                quant=m.get("quant", "Q4_K_M"),
                tags=m.get("tags", ()),
                repo_id=m.get("repo_id", ""),
            )
            row["download_bytes"] = download_bytes
            row["download_bytes_estimated"] = True
            row["download_available"] = True
        else:
            download_bytes = 0
            row["download_available"] = False

        if download_bytes > 0:
            row["hardware_note"] = _format_catalog_note(
                est_vram_gb=fit["est_vram_mb"] / 1024,
                download_bytes=download_bytes,
                headroom_gb=headroom_gb,
                fit=fit["hardware_fit"],
                tier=tier,
            )
        enriched.append(row)

    enriched.sort(
        key=lambda m: (
            -m.get("hardware_fit_rank", 0),
            -(m.get("priority") or 0),
            m.get("name", ""),
        )
    )
    if diversify:
        enriched = diversify_by_family(enriched)
        indexed = list(enumerate(enriched))
        indexed.sort(
            key=lambda m: (
                -m[1].get("hardware_fit_rank", 0),
                m[0],
            )
        )
        enriched = [m for _, m in indexed]
    return enriched


def _guess_params_from_name(name: str) -> float | None:
    m = _PARAM_RE.search(name)
    return float(m.group(1)) if m else None


def assess_inference_option_fit(option: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    size_bytes = int(option.get("size_bytes") or 0)
    name = option.get("name") or ""
    if size_bytes > 0:
        est_gb = round(size_bytes / (1024**3) + 0.8, 2)
    else:
        guessed = _guess_params_from_name(name)
        if guessed:
            est_gb = estimate_chat_vram_gb(f"{guessed}B")
        elif option.get("kind") == "ollama":
            guessed = _guess_params_from_name(name.split(":")[0])
            est_gb = estimate_chat_vram_gb(f"{guessed or 7}B") if guessed else 5.0
        else:
            est_gb = 6.0
    return assess_hardware_fit(est_gb, profile)


def preferred_inference_backend(profile: dict[str, Any]) -> str:
    tier = classify_tier(profile)
    headroom = vram_headroom_mb(profile)
    try:
        backend = Backend(profile.get("backend", "cpu"))
    except ValueError:
        backend = Backend.CPU

    # Quantized GGUF via llama.cpp is best on CPU-only or tight unified/discrete memory.
    if tier == HardwareTier.CPU_ONLY:
        return "llamacpp"
    if headroom < 6000 or tier == HardwareTier.EDGE:
        return "llamacpp"
    if tier == HardwareTier.APPLE_UNIFIED and headroom < 16384:
        return "llamacpp"
    if tier == HardwareTier.APPLE_UNIFIED or backend == Backend.MLX:
        return "mlx"
    return "llamacpp"


def memory_headroom_label(profile: dict[str, Any]) -> str:
    """Human label for free memory (RAM on Apple/CPU, VRAM on discrete GPU)."""
    tier = classify_tier(profile)
    if tier in (HardwareTier.APPLE_UNIFIED, HardwareTier.CPU_ONLY):
        return "RAM"
    return "VRAM"


def training_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    from seiso.training.platform_caps import training_capabilities

    tier = classify_tier(profile)
    headroom = vram_headroom_mb(profile)
    ram = float(profile.get("ram_gb") or 0)
    caps = training_capabilities()

    if tier in (HardwareTier.WORKSTATION, HardwareTier.CAPABLE) and headroom >= 16000:
        batch, accum, max_seq, max_params = 2, 4, 4096, "14B"
    elif headroom >= 10000 or tier == HardwareTier.APPLE_UNIFIED:
        batch, accum, max_seq, max_params = 1, 8, 2048, "7B"
    elif headroom >= 6000:
        batch, accum, max_seq, max_params = 1, 16, 2048, "3B"
    else:
        batch, accum, max_seq, max_params = 1, 16, 1024, "1B"

    quant = caps["recommended_quant"]
    note = f"Tuned for {TIER_LABELS[tier]} ({ram:.0f} GB RAM, ~{headroom // 1024} GB free)"
    if not caps["supports_qlora"]:
        note += " — use 16-bit LoRA on macOS (no bitsandbytes)"
    if caps["fused_kernels_available"]:
        note += f" — fused kernels via {caps['kernel_backend']}"

    return {
        "batch_size": batch,
        "gradient_accumulation_steps": accum,
        "max_seq_length": max_seq,
        "quant": quant,
        "method": "lora",
        "gradient_checkpointing": True,
        "max_recommended_params": max_params,
        "use_fused_kernels": caps["fused_kernels_available"],
        "use_fused_ce": caps["fused_ce_available"],
        "kernel_backend": caps["kernel_backend"],
        "train_platform": caps["train_platform"],
        "multi_gpu_available": caps["multi_gpu_available"],
        "note": note,
    }


def recommended_catalog_repo(profile: dict[str, Any], *, task: str = "chat") -> str | None:
    from seiso.models.catalog import search_catalog

    models = search_catalog(task=task) if task else search_catalog()
    models = enrich_catalog_models(models, profile, fetch_sizes=False, diversify=True)
    for m in models:
        if m.get("hardware_fit") in ("ideal", "good") and m.get("task") != "embedding":
            return m["repo_id"]
    for m in models:
        if m.get("task") != "embedding":
            return m["repo_id"]
    return None


def enrich_profile(profile: dict[str, Any]) -> dict[str, Any]:
    tier = classify_tier(profile)
    headroom = vram_headroom_mb(profile)
    recommended_chat = recommended_catalog_repo(profile, task="chat")
    return {
        **profile,
        "tier": tier.value,
        "tier_label": TIER_LABELS[tier],
        "effective_vram_mb": effective_budget_mb(profile),
        "vram_headroom_mb": headroom,
        "preferred_inference_backend": preferred_inference_backend(profile),
        "training_defaults": training_defaults(profile),
        "recommended_chat_repo": recommended_chat,
        "recommended_train_repo": recommended_chat,
    }


def hardware_summary(profile: dict[str, Any]) -> dict[str, Any]:
    """Compact summary safe to embed in API responses."""
    tier = classify_tier(profile)
    preferred = preferred_inference_backend(profile)
    return {
        "tier": tier.value,
        "tier_label": TIER_LABELS[tier],
        "backend": profile.get("backend"),
        "ram_gb": profile.get("ram_gb"),
        "gpu_count": len(profile.get("gpus") or []),
        "effective_vram_mb": effective_budget_mb(profile),
        "vram_headroom_mb": vram_headroom_mb(profile),
        "memory_headroom_label": memory_headroom_label(profile),
        "preferred_inference_backend": preferred,
        "preferred_inference_backend_label": INFERENCE_BACKEND_LABELS.get(preferred, preferred),
        "local_only": True,
    }
