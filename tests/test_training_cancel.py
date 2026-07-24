"""Tests for cooperative training cancellation."""

from __future__ import annotations

from seiso.training.cancel import clear, is_requested, register, request


def test_training_cancel_register_and_request():
    register("job-1")
    assert not is_requested("job-1")
    request("job-1")
    assert is_requested("job-1")
    clear("job-1")
    assert not is_requested("job-1")


def test_training_cancel_before_register_is_sticky():
    request("job-early")
    assert is_requested("job-early")
    register("job-early")
    assert is_requested("job-early")
    clear("job-early")
    assert not is_requested("job-early")
