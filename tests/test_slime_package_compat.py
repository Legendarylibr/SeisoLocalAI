"""Compatibility package and naming aliases for slime."""

from __future__ import annotations


def test_slime_package_exports_preferred_and_legacy_names():
    from seiso.slime import SingleGpuSlimeConfig, SlimeConfig
    from seiso.slime.trainer import train_single_gpu_slime, train_slime

    assert SlimeConfig is SingleGpuSlimeConfig
    assert train_slime is train_single_gpu_slime


def test_slime_single_gpu_shim_reexports():
    from seiso.slime.config import SingleGpuSlimeConfig as New
    from seiso.slime.trainer import train_slime
    from seiso.slime_single_gpu.config import SingleGpuSlimeConfig as Old
    from seiso.slime_single_gpu.trainer import train_single_gpu_slime

    assert Old is New
    assert train_single_gpu_slime is train_slime


def test_metrics_filename_unchanged_for_compat():
    """Existing dashboards and Forge look for this metrics file name."""
    import inspect

    from seiso.slime import trainer as t

    src = inspect.getsource(t)
    assert "slime_single_gpu_metrics.jsonl" in src
