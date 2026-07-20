"""Tests for code corpus normalization."""

from __future__ import annotations

from seiso.training.code_corpus import (
    is_metadata_only_row,
    normalize_code_row,
    recommend_pretraining_epochs,
)


def test_normalize_code_row_with_language_and_path():
    sample = normalize_code_row(
        {
            "language": "Python",
            "repo": "org/repo",
            "rel_path": "src/main.py",
            "text": "print('hi')",
        },
        source="org/repo",
    )
    assert sample is not None
    assert "# language: python" in sample.text
    assert "print('hi')" in sample.text


def test_normalize_code_row_preserves_indent_and_code_column():
    body = "    def add(a, b):\n        return a + b\n"
    sample = normalize_code_row({"code": body, "language": "Python"})
    assert sample is not None
    assert "    def add(a, b):\n        return a + b" in sample.text
    assert sample.text.startswith("# language: python\n")


def test_is_metadata_only_row():
    assert is_metadata_only_row({"repo": "x", "rel_path": "a.py"})
    assert not is_metadata_only_row({"repo": "x", "text": "code body here " * 4})
    assert not is_metadata_only_row({"repo": "x", "code": "def f():\n    pass\n"})


def test_recommend_pretraining_epochs_large_corpus():
    info = recommend_pretraining_epochs(
        sample_count=2_000_000,
        max_seq_length=1024,
        model_params_b=7.0,
    )
    assert info["recommended_epochs"] == 1
