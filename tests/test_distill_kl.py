"""Unit tests for shifted+masked distillation KL and vocab checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from seiso.codellama_compress.distill import (
    assert_compatible_teacher_student,
    shifted_masked_kl_div,
)


def test_assert_compatible_teacher_student_rejects_vocab_mismatch():
    teacher = SimpleNamespace(config=SimpleNamespace(vocab_size=100))
    student = SimpleNamespace(config=SimpleNamespace(vocab_size=200))
    with pytest.raises(ValueError, match="vocab_size"):
        assert_compatible_teacher_student(teacher, student)


def test_assert_compatible_teacher_student_accepts_match():
    teacher = SimpleNamespace(config=SimpleNamespace(vocab_size=128))
    student = SimpleNamespace(config=SimpleNamespace(vocab_size=128))
    assert_compatible_teacher_student(teacher, student)


def test_shifted_masked_kl_ignores_padding_and_uses_next_token_positions():
    # Batch=1, seq=4, vocab=3. Pad last two positions.
    # Student matches teacher on real tokens → KL near 0 on masked mean.
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    kl = shifted_masked_kl_div(logits, logits, mask, temperature=1.0)
    assert float(kl.item()) == pytest.approx(0.0, abs=1e-5)

    # Mismatch only on a padded prediction slot must not increase masked KL.
    bad_pad = logits.clone()
    bad_pad[0, 2] = torch.tensor([0.0, 0.0, 10.0])
    kl_pad = shifted_masked_kl_div(logits, bad_pad, mask, temperature=1.0)
    assert float(kl_pad.item()) == pytest.approx(0.0, abs=1e-5)

    # Mismatch on a real next-token position increases KL.
    bad_real = logits.clone()
    bad_real[0, 0] = torch.tensor([0.0, 10.0, 0.0])
    kl_real = shifted_masked_kl_div(logits, bad_real, mask, temperature=1.0)
    assert float(kl_real.item()) > 1.0
