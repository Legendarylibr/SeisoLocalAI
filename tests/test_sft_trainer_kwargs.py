"""SFT trainer construction must not pass seiso-only kwargs to plain TRL SFTTrainer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from seiso.training import sft as sft_mod


def test_build_sft_trainer_strips_seiso_kwargs_for_plain_sft():
    """use_fused_ce=False must not pass use_cuda_graphs/use_fused_ce to TRL SFTTrainer."""
    plain = MagicMock(name="SFTTrainer")
    plain.__name__ = "SFTTrainer"
    plain.return_value = SimpleNamespace(name="plain")
    fused = MagicMock(name="FusedSFTTrainer")
    fused.__name__ = "FusedSFTTrainer"
    fused.return_value = SimpleNamespace(name="fused")

    with (
        patch.object(sft_mod, "_SFTTrainer", plain),
        patch.object(sft_mod, "FusedSFTTrainer", fused),
        patch.object(sft_mod, "SFTConfig", MagicMock(return_value=SimpleNamespace())),
        patch.object(sft_mod, "_sft_max_length_key", return_value="max_length"),
    ):
        out = sft_mod.build_sft_trainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_ds=MagicMock(),
            eval_ds=None,
            training_args_dict={"output_dir": "/tmp", "num_train_epochs": 1},
            max_seq_length=128,
            use_fused_ce=False,
            use_cuda_graphs=False,
        )

    assert out.name == "plain"
    plain.assert_called_once()
    fused.assert_not_called()
    kwargs = plain.call_args.kwargs
    assert "use_fused_ce" not in kwargs
    assert "use_cuda_graphs" not in kwargs


def test_build_sft_trainer_uses_fused_class_when_fused_ce():
    plain = MagicMock(name="SFTTrainer")
    plain.__name__ = "SFTTrainer"
    fused = MagicMock(name="FusedSFTTrainer")
    fused.__name__ = "FusedSFTTrainer"
    fused.return_value = SimpleNamespace(name="fused")

    with (
        patch.object(sft_mod, "_SFTTrainer", plain),
        patch.object(sft_mod, "FusedSFTTrainer", fused),
        patch.object(sft_mod, "SFTConfig", MagicMock(return_value=SimpleNamespace())),
        patch.object(sft_mod, "_sft_max_length_key", return_value="max_length"),
    ):
        out = sft_mod.build_sft_trainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_ds=MagicMock(),
            eval_ds=None,
            training_args_dict={"output_dir": "/tmp", "num_train_epochs": 1},
            max_seq_length=128,
            use_fused_ce=True,
            use_cuda_graphs=False,
        )

    assert out.name == "fused"
    fused.assert_called_once()
    kwargs = fused.call_args.kwargs
    assert kwargs["use_fused_ce"] is True
    assert kwargs["use_cuda_graphs"] is False


def test_fused_trainer_tokenizer_fallback_keeps_seiso_kwargs():
    """processing_class TypeError must not drop use_fused_ce/use_cuda_graphs on FusedSFTTrainer."""
    plain = MagicMock(name="SFTTrainer")
    plain.__name__ = "SFTTrainer"
    fused = MagicMock(name="FusedSFTTrainer")
    fused.__name__ = "FusedSFTTrainer"
    # First call (processing_class=) fails like older TRL; second (tokenizer=) succeeds.
    fused.side_effect = [
        TypeError("unexpected keyword argument 'processing_class'"),
        SimpleNamespace(name="fused"),
    ]

    with (
        patch.object(sft_mod, "_SFTTrainer", plain),
        patch.object(sft_mod, "FusedSFTTrainer", fused),
        patch.object(sft_mod, "SFTConfig", MagicMock(return_value=SimpleNamespace())),
        patch.object(sft_mod, "_sft_max_length_key", return_value="max_length"),
    ):
        out = sft_mod.build_sft_trainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_ds=MagicMock(),
            eval_ds=None,
            training_args_dict={"output_dir": "/tmp", "num_train_epochs": 1},
            max_seq_length=128,
            use_fused_ce=False,
            use_cuda_graphs=True,  # selects FusedSFTTrainer; must survive tokenizer= retry
        )

    assert out.name == "fused"
    assert fused.call_count == 2
    retry_kwargs = fused.call_args_list[1].kwargs
    assert "processing_class" not in retry_kwargs
    assert "tokenizer" in retry_kwargs
    assert retry_kwargs["use_fused_ce"] is False
    assert retry_kwargs["use_cuda_graphs"] is True
