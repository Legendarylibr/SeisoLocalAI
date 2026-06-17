"""Platform capability and training loader tests."""

from seiso.training.platform_caps import training_capabilities


def test_training_capabilities_shape():
    caps = training_capabilities()
    assert "os" in caps
    assert "train_platform" in caps
    assert "supports_qlora" in caps
    assert "fused_kernels_available" in caps
    assert "recommended_quant" in caps
    assert caps["recommended_quant"] in ("4bit", "16bit", "8bit", "none")


def test_training_loader_skips_mlx(monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    from unittest.mock import MagicMock, patch

    from seiso.models.loader import LoadOptions, ModelKind, load_model

    opts = LoadOptions(model_id="test/model", kind=ModelKind.TEXT)

    with patch("seiso.models.loader.load_mlx") as mlx_load:
        with patch("seiso.models.torch_loader.load_torch") as torch_load:
            torch_load.return_value = (MagicMock(), MagicMock())
            load_model(opts, for_training=True)
            mlx_load.assert_not_called()
            torch_load.assert_called_once()
