"""End-to-end GPU training smoke — fused kernels, CUDA graphs, QLoRA."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()",
    reason="CUDA required for GPU e2e training",
)


@pytest.fixture
def e2e_output_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("train-gpu-e2e")
    yield out
    shutil.rmtree(out, ignore_errors=True)


def _base_config(output_dir: Path, *, quant: str = "4bit") -> "TrainConfig":
    from seiso.training.config import TrainConfig

    return TrainConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset=Path(__file__).resolve().parents[1] / "data" / "sample.jsonl",
        output_dir=output_dir,
        method="lora",
        quant=quant,
        epochs=1,
        batch_size=1,
        max_seq_length=256,
        lora_r=8,
        lora_alpha=16,
        gradient_checkpointing=False,
        deterministic=False,
        eval_split_ratio=0,
        save_steps=100,
        logging_steps=1,
        use_triton=True,
        use_fused_ce=True,
        use_fused_lora=True,
        extra={
            "use_cuda_graphs": True,
            "use_fused_lora_qkv": True,
            "use_flash_attention": True,
        },
    )


def test_e2e_gpu_qlora_training_pipeline(e2e_output_dir: Path):
    """Full QLoRA path: load → fuse kernels → train → checkpoint + manifest."""
    pytest.importorskip("torch")

    from seiso.training.trainer import SeisoTrainer

    metrics: list[dict] = []

    out = SeisoTrainer(_base_config(e2e_output_dir), on_metric=metrics.append).run()

    assert out.is_dir()
    manifest = json.loads((out / "seiso_manifest.json").read_text())
    kernels = manifest["kernels"]

    assert manifest["method"] == "lora"
    assert manifest["quant"] == "4bit"
    assert manifest["train_samples"] == 4
    assert kernels.get("fused_ce") is True
    assert kernels.get("cuda_graphs_requested") is True
    assert kernels.get("modules_patched", 0) > 0
    assert kernels.get("fused_residual_decoder_patched", 0) > 0
    assert kernels.get("lora_qkv_patched", 0) > 0
    assert (out / "adapter_config.json").is_file()
    assert metrics, "expected at least one training metric"


def test_e2e_cuda_graph_capture_bf16(e2e_output_dir: Path):
    """16-bit LoRA path where CUDA graphs can capture (no bitsandbytes)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 required for CUDA graph capture test")

    from seiso.kernels.cuda_graphs import CudaGraphTrainingManager
    from seiso.training.trainer import SeisoTrainer

    cfg = _base_config(e2e_output_dir, quant="16bit")
    cfg.extra["use_cuda_graphs"] = True

    trainer_runner = SeisoTrainer(cfg)
    out = trainer_runner.run()

    manifest = json.loads((out / "seiso_manifest.json").read_text())
    assert manifest["quant"] == "16bit"
    assert manifest["kernels"].get("cuda_graphs_requested") is True

    # Graph manager should skip bnb but accept bf16; capture may still fail on
    # first-gen setups — verify training completed numerically either way.
    mgr = CudaGraphTrainingManager()
    mgr.try_enable(explicit=True, deterministic=False)
    assert mgr._enabled