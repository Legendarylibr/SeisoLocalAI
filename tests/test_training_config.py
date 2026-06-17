from seiso.training.config import QuantMode, TrainConfig, TrainMethod


def test_train_config_from_dict():
    cfg = TrainConfig.model_validate({
        "model_id": "test/model",
        "dataset": "./update.jsonl",
        "method": "lora",
        "quant": "4bit",
    })
    assert cfg.method == TrainMethod.LORA
    assert cfg.quant == QuantMode.INT4
    assert cfg.lora_r == 16
