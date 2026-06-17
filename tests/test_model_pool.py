"""Tests for VRAM model pool."""

from seiso.inference.model_pool import ModelPool


def test_pool_singleton():
    a = ModelPool.get()
    b = ModelPool.get()
    assert a is b


def test_unload_clears_active():
    pool = ModelPool.get()
    pool.unload_all()
    assert pool.active_key is None
    status = pool.status()
    assert status["active_model"] is None


def test_generation_invalidation():
    pool = ModelPool.get()
    gen_a = pool.bump_generation()
    assert pool.is_generation_active(gen_a)
    gen_b = pool.bump_generation()
    assert not pool.is_generation_active(gen_a)
    assert pool.is_generation_active(gen_b)


def test_cancel_and_unload_clears_active():
    pool = ModelPool.get()
    pool.cancel_and_unload()
    assert pool.active_key is None
