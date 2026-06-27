"""Tests for modern training practice helpers."""

from __future__ import annotations

from seiso.training.config import DatasetFormat, TrainConfig, TrainMethod
from seiso.training.practices import (
    default_dataloader_num_workers,
    default_dataloader_prefetch_factor,
    default_dataset_num_proc,
    learning_rate_for_method,
    resolve_compute_dtype,
    resolve_dataloader_settings,
    resolve_optimizer,
    sft_modern_kwargs,
    warmup_ratio_for_corpus,
)


def test_learning_rate_scales_with_method():
    assert learning_rate_for_method(TrainMethod.LORA) == 2e-4
    assert learning_rate_for_method(TrainMethod.FULL) == 1e-5


def test_warmup_ratio_scales_with_corpus():
    assert warmup_ratio_for_corpus(100) == 0.1
    assert warmup_ratio_for_corpus(2_000) == 0.05
    assert warmup_ratio_for_corpus(20_000) == 0.03


def test_resolve_compute_dtype_prefers_bf16_on_cuda():
    use_bf16, use_fp16 = resolve_compute_dtype(
        cuda_available=True,
        bf16_supported=True,
        quant="4bit",
    )
    assert use_bf16 is True
    assert use_fp16 is False


def test_sft_modern_kwargs_chat_assistant_only():
    cfg = TrainConfig.model_validate(
        {
            "model_id": "org/model",
            "dataset": "data.jsonl",
            "dataset_num_proc": 2,
            "pad_to_multiple_of": 16,
        }
    )
    kwargs = sft_modern_kwargs(
        cfg,
        train_on_responses_only=True,
        dataset_format=DatasetFormat.CHAT,
        cuda_available=True,
        use_text_field=True,
    )
    assert kwargs["dataset_num_proc"] == 2
    assert kwargs["pad_to_multiple_of"] == 16
    assert kwargs["assistant_only_loss"] is True


def test_default_dataset_num_proc_disabled_when_zero():
    assert default_dataset_num_proc(0) is None


def test_default_dataloader_workers_overlap_cuda_input_pipeline():
    assert default_dataloader_num_workers(0, cuda_available=True, cpu_count=12) == 4
    assert default_dataloader_prefetch_factor(None, num_workers=4, cuda_available=True) == 2


def test_default_dataloader_workers_stay_single_process_on_cpu():
    assert default_dataloader_num_workers(0, cuda_available=False, cpu_count=12) == 0
    assert default_dataloader_prefetch_factor(None, num_workers=0, cuda_available=False) is None


def test_resolve_dataloader_settings_honors_explicit_prefetch():
    cfg = TrainConfig.model_validate(
        {
            "model_id": "org/model",
            "dataset": "data.jsonl",
            "dataloader_num_workers": 2,
            "dataloader_prefetch_factor": 4,
        }
    )
    assert resolve_dataloader_settings(cfg, cuda_available=True) == (2, True, 4)


def test_resolve_dataloader_settings_disables_persistent_without_workers():
    cfg = TrainConfig.model_validate(
        {
            "model_id": "org/model",
            "dataset": "data.jsonl",
            "dataloader_persistent_workers": True,
        }
    )
    assert resolve_dataloader_settings(cfg, cuda_available=False) == (0, False, None)


def test_resolve_optimizer_cpu():
    assert resolve_optimizer("4bit", use_cpu=True) == "adamw_torch"
