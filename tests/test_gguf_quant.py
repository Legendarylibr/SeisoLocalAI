"""Tests for shared GGUF quant parsing."""

from __future__ import annotations

from seiso.models.gguf_quant import (
    effective_bits_for_quant,
    extract_quant_label_from_text,
    normalize_quant_label,
    rank_gguf_filenames,
)


def test_normalize_quant_label():
    assert normalize_quant_label("q4-k-xl") == "Q4_K_XL"
    assert normalize_quant_label("UD-Q4") == "UD_Q4"


def test_extract_quant_label_from_text():
    assert extract_quant_label_from_text("Model-Q4_K_M.gguf") == "Q4_K_M"
    assert extract_quant_label_from_text("weights-IQ4_XS.gguf") == "IQ4_XS"
    assert extract_quant_label_from_text("model-Q4_K_XL.gguf") == "Q4_K_XL"
    assert extract_quant_label_from_text("model-f16.gguf") == "F16"


def test_effective_bits_for_known_and_unknown():
    assert effective_bits_for_quant("Q4_K_M") == 4.83
    assert effective_bits_for_quant("Q4_K_XL") == 4.83
    assert 4.0 < effective_bits_for_quant("UD_Q4") < 5.0


def test_rank_gguf_filenames_prefers_without_excluding():
    files = ["model-Q8_0.gguf", "model-Q4_K_XL.gguf", "model-Q4_K_M.gguf"]
    assert rank_gguf_filenames(files, preferred="Q4_K_XL")[0] == "model-Q4_K_XL.gguf"
    assert set(rank_gguf_filenames(files)) == set(files)
