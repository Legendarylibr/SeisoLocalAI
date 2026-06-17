from __future__ import annotations

# PyTorch 2.12+ binary matrix (see https://pytorch.org/get-started/locally/):
# - cu130: stable default for current NVIDIA drivers
# - cu126: legacy fallback for older drivers / architectures
# - cu128: removed in PyTorch 2.12 — do not reference in install docs
TORCH_CUDA_INDEX_CU130 = "https://download.pytorch.org/whl/cu130"
TORCH_CUDA_INDEX_CU126 = "https://download.pytorch.org/whl/cu126"
DEFAULT_CUDA_INDEX = TORCH_CUDA_INDEX_CU130
INSTALL_CUDA_TORCH_SCRIPT = "python3 scripts/install_cuda_torch.py"


def cuda_torch_pip_argv(
    *,
    python: str,
    index_url: str = DEFAULT_CUDA_INDEX,
    force_reinstall: bool = False,
) -> list[str]:
    """Return argv for installing a CUDA-enabled torch wheel."""
    cmd = [python, "-m", "pip", "install"]
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd.extend(["--upgrade", "torch", "--index-url", index_url])
    return cmd


def cuda_torch_pip_command(*, index_url: str = DEFAULT_CUDA_INDEX) -> str:
    """Return a pip command that installs a CUDA-enabled torch wheel."""
    return " ".join(cuda_torch_pip_argv(python="python3", index_url=index_url))


def cuda_torch_install_instructions(*, index_url: str = DEFAULT_CUDA_INDEX) -> str:
    """Multi-line guidance for fixing a CPU-only or mismatched PyTorch install."""
    legacy = TORCH_CUDA_INDEX_CU126
    return (
        "Install a CUDA-enabled PyTorch wheel before GPU training:\n"
        f"  {INSTALL_CUDA_TORCH_SCRIPT}\n"
        "Or install manually:\n"
        f"  {cuda_torch_pip_command(index_url=index_url)}\n"
        f"  python3 -m pip install -e .\n"
        "If that wheel does not match your driver, try the legacy CUDA 12.6 build:\n"
        f"  {cuda_torch_pip_command(index_url=legacy)}\n"
        "Verify with:\n"
        "  python3 scripts/install_cuda_torch.py --check-only"
    )


def torch_cuda_ready_report() -> dict[str, object]:
    """Canonical JSON-friendly summary of the active torch/CUDA install."""
    from adaptive_quant.hardware import nvidia_smi_visible, resolve_nvidia_smi_executable
    from adaptive_quant.torch_policy import torch_cuda_diagnostics

    report: dict[str, object] = dict(torch_cuda_diagnostics("cuda"))
    smi_visible = nvidia_smi_visible()
    report["nvidia_smi_visible"] = smi_visible
    report["nvidia_smi_path"] = resolve_nvidia_smi_executable()

    if not report.get("torch_installed", False):
        report["policy_gradient_probe"] = {"ok": False, "error": "torch_not_installed"}
        report.setdefault("install_hint", INSTALL_CUDA_TORCH_SCRIPT)
        return report

    probe_device = "cuda" if report.get("cuda_available") else "cpu"
    probe = probe_torch_policy_gradient_flow(device=str(probe_device))
    report["policy_gradient_probe"] = probe
    if not probe.get("ok"):
        report["policy_gradient_probe_failed"] = True

    cuda_version = report.get("cuda_version")
    if not report.get("cuda_available"):
        if cuda_version is None:
            report["likely_cpu_only_wheel"] = True
        report.setdefault("install_hint", INSTALL_CUDA_TORCH_SCRIPT)
        if report.get("likely_cpu_only_wheel") and smi_visible:
            report["driver_gpu_detected"] = True
        return report

    return report


def probe_torch_policy_gradient_flow(*, device: str = "cpu") -> dict[str, object]:
    """Run a tiny policy forward+backward to catch tensor-alias bugs (e.g. expand views)."""
    from adaptive_quant.configuration import FrameworkConfig
    from adaptive_quant.torch_policy import TorchActorCritic, torch

    if torch is None:
        return {"ok": False, "error": "torch_not_installed"}

    config = FrameworkConfig(
        training_backend="pytorch",
        torch_device=device,
        torch_compile=False,
        torch_hidden_dim=16,
        torch_mlp_depth=1,
        stability_probe_count=1,
        run_name="policy_gradient_probe",
    )
    resolved = device.strip().lower()
    if resolved == "cuda" and not torch.cuda.is_available():
        resolved = "cpu"

    try:
        model = TorchActorCritic(config)
        target = torch.device(resolved)
        model = model.to(target)
        model.train()
        batch_size = 4
        states = torch.randn(batch_size, config.state_vector_dim(), device=target)
        outputs = model(states)
        loss = outputs["learned_std"].float().pow(2).mean()
        loss = loss + outputs["learned_mean"].float().pow(2).mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        return {
            "ok": True,
            "device": str(target),
            "batch_size": batch_size,
        }
    except RuntimeError as exc:
        message = str(exc)
        hint = None
        if "single memory location" in message.lower():
            hint = (
                "Policy tensors alias the same storage (often from expand/view). "
                "Update the repo to a build that uses repeat/clone for broadcast heads."
            )
        return {
            "ok": False,
            "device": resolved,
            "error": message,
            "error_type": type(exc).__name__,
            "hint": hint,
        }


def validate_cuda_after_install(
    requested_device: str = "cuda",
    *,
    report: dict[str, object] | None = None,
) -> None:
    """Raise when CUDA is unavailable or the active wheel cannot run the visible GPU."""
    active = report if report is not None else torch_cuda_ready_report()
    report = active
    if not report.get("cuda_available"):
        hint = report.get("install_hint") or INSTALL_CUDA_TORCH_SCRIPT
        smi = " nvidia-smi sees a GPU but" if report.get("driver_gpu_detected") else ""
        raise RuntimeError(
            f"CUDA is not available after installing torch.{smi} "
            f"Confirm the driver with `nvidia-smi`, then retry:\n  {hint}"
        )
    from adaptive_quant.torch_policy import validate_cuda_runtime_compatibility

    validate_cuda_runtime_compatibility(requested_device)
    probe = report.get("policy_gradient_probe")
    if isinstance(probe, dict) and not probe.get("ok"):
        hint = probe.get("hint") or (
            "Policy gradient probe failed. Run `python3 scripts/install_cuda_torch.py --check-only` "
            "for details and update the repo if the error mentions tensor memory aliasing."
        )
        raise RuntimeError(
            "Torch policy backward probe failed before GPU training.\n"
            f"  device: {probe.get('device', requested_device)}\n"
            f"  error: {probe.get('error')}\n"
            f"  hint: {hint}"
        )
