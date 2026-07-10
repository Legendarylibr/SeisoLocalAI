"""Tests for PyTorch model loading options."""

from __future__ import annotations

from types import SimpleNamespace

from seiso.models.loader import Backend, LoadOptions


def test_resolve_device_map_honors_explicit_balanced(monkeypatch):
    from seiso.models import torch_loader

    monkeypatch.setattr(torch_loader, "_cuda_available", lambda: True)

    assert (
        torch_loader._resolve_device_map(Backend.TORCH, requested="balanced_low_0")
        == "balanced_low_0"
    )


def test_resolve_device_map_can_disable_auto(monkeypatch):
    from seiso.models import torch_loader

    monkeypatch.setattr(torch_loader, "_cuda_available", lambda: True)

    assert torch_loader._resolve_device_map(Backend.TORCH, requested="off") is None


def test_load_torch_passes_auto_device_map_and_max_memory(monkeypatch):
    from seiso.models import torch_loader

    model_calls: list[dict] = []
    tokenizer_calls: list[dict] = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            return SimpleNamespace(quantization_config=None)

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __len__(self) -> int:
            return 8

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            tokenizer_calls.append(kwargs)
            return FakeTokenizer()

    class FakeEmbeddings:
        weight = SimpleNamespace(shape=(8, 4))

    class FakeModel:
        def get_input_embeddings(self):
            return FakeEmbeddings()

        def resize_token_embeddings(self, _size: int) -> None:
            raise AssertionError("tokenizer and embeddings should already match")

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            model_calls.append(kwargs)
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoConfig=FakeAutoConfig,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoTokenizer=FakeAutoTokenizer,
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setattr(torch_loader, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        "seiso.kernels.attention.resolve_attention_implementation",
        lambda prefer_fa3=True: "sdpa",
    )
    monkeypatch.setattr(
        "seiso.memory.protection.build_hf_max_memory",
        lambda: {0: "10GiB", "cpu": "32GiB"},
    )

    torch_loader.load_torch(
        LoadOptions(model_id="org/model", device_map="balanced_low_0"),
        backend=Backend.TORCH,
    )

    assert tokenizer_calls[0]["revision"] == "main"
    assert model_calls[0]["device_map"] == "balanced_low_0"
    assert model_calls[0]["max_memory"] == {0: "10GiB", "cpu": "32GiB"}
    assert model_calls[0]["attn_implementation"] == "sdpa"


def test_load_torch_allows_disabling_device_map(monkeypatch):
    from seiso.models import torch_loader

    model_calls: list[dict] = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return SimpleNamespace(quantization_config=None)

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __len__(self) -> int:
            return 8

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return FakeTokenizer()

    class FakeEmbeddings:
        weight = SimpleNamespace(shape=(8, 4))

    class FakeModel:
        def get_input_embeddings(self):
            return FakeEmbeddings()

        def resize_token_embeddings(self, _size: int) -> None:
            raise AssertionError("tokenizer and embeddings should already match")

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            model_calls.append(kwargs)
            return FakeModel()

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=FakeAutoConfig,
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
        ),
    )
    monkeypatch.setattr(torch_loader, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        "seiso.kernels.attention.resolve_attention_implementation",
        lambda prefer_fa3=True: "sdpa",
    )

    torch_loader.load_torch(
        LoadOptions(model_id="org/model", device_map="off"),
        backend=Backend.TORCH,
    )

    assert "device_map" not in model_calls[0]
    assert "max_memory" not in model_calls[0]


def test_load_torch_retries_without_bitsandbytes_quantization(monkeypatch):
    from seiso.models import torch_loader

    model_calls: list[dict] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        float16=object(),
        bfloat16=object(),
    )

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return SimpleNamespace(quantization_config=None)

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __len__(self) -> int:
            return 8

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return FakeTokenizer()

    class FakeEmbeddings:
        weight = SimpleNamespace(shape=(8, 4))

    class FakeModel:
        def get_input_embeddings(self):
            return FakeEmbeddings()

        def resize_token_embeddings(self, _size: int) -> None:
            raise AssertionError("tokenizer and embeddings should already match")

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            model_calls.append(kwargs)
            if len(model_calls) == 1:
                raise RuntimeError("bitsandbytes rejected this quantization_config")
            return FakeModel()

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=FakeAutoConfig,
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
        ),
    )
    monkeypatch.setattr(torch_loader, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        "seiso.kernels.attention.resolve_attention_implementation",
        lambda prefer_fa3=True: "sdpa",
    )
    monkeypatch.setattr("seiso.memory.protection.build_hf_max_memory", lambda: None)

    torch_loader.load_torch(
        LoadOptions(model_id="org/model", load_in_4bit=True),
        backend=Backend.TORCH,
    )

    assert len(model_calls) == 2
    assert "quantization_config" in model_calls[0]
    assert "quantization_config" not in model_calls[1]
    assert model_calls[1]["device_map"] == "auto"


def test_load_torch_falls_back_from_flash_attention_to_sdpa(monkeypatch):
    from seiso.models import torch_loader

    model_calls: list[str] = []

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return SimpleNamespace(quantization_config=None)

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __len__(self) -> int:
            return 8

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            return FakeTokenizer()

    class FakeModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(shape=(8, 4)))

        def resize_token_embeddings(self, _size: int) -> None:
            raise AssertionError("tokenizer and embeddings should match")

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            implementation = kwargs["attn_implementation"]
            model_calls.append(implementation)
            if implementation == "flash_attention_2":
                raise RuntimeError("flash_attn is not supported for this model")
            return FakeModel()

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=FakeAutoConfig,
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
        ),
    )
    monkeypatch.setattr(torch_loader, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        "seiso.kernels.attention.resolve_attention_implementation",
        lambda prefer_fa3=True: "flash_attention_2",
    )
    monkeypatch.setattr("seiso.memory.protection.build_hf_max_memory", lambda: None)
    monkeypatch.setattr(
        "seiso.memory.protection.release_cached_memory", lambda **_kwargs: None
    )

    model, _tokenizer = torch_loader.load_torch(
        LoadOptions(model_id="org/model"),
        backend=Backend.TORCH,
    )

    assert model_calls == ["flash_attention_2", "sdpa"]
    assert model._seiso_attention_implementation == "sdpa"
