"""CUDA graph training manager tests (no GPU capture required)."""

from seiso.kernels.cuda_graphs import (
    CudaGraphTrainingManager,
    cuda_graphs_enabled,
)


def test_cuda_graphs_enabled_respects_deterministic():
    assert cuda_graphs_enabled(explicit=True, deterministic=True) is False


def test_cuda_graph_manager_metadata():
    mgr = CudaGraphTrainingManager()
    mgr.try_enable(explicit=False, deterministic=True)
    assert mgr.active is False
    meta = mgr.metadata()
    assert "cuda_graphs_enabled" in meta
    assert meta["cuda_graphs_captured"] is False


def test_shape_key_stable():
    class FakeTensor:
        shape = (2, 128)
        dtype = "torch.int64"
        device = type("D", (), {"type": "cuda"})()

    key = CudaGraphTrainingManager._shape_key({"input_ids": FakeTensor()})
    assert key == (("input_ids", (2, 128), "torch.int64", "cuda"),)
