"""Tests for modern training practice helpers."""

from __future__ import annotations

from seiso.training.config import DatasetFormat, TrainConfig, TrainMethod
from seiso.training.practices import (
    default_dataset_num_proc,
    learning_rate_for_method,
    resolve_compute_dtype,
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


def test_resolve_optimizer_cpu():
    assert resolve_optimizer("4bit", use_cpu=True) == "adamw_torch"