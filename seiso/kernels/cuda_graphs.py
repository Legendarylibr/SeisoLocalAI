"""CUDA Graph capture for steady-state training — forward + backward replay."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_WARMUP_STEPS = 3
_CAPTURE_WARMUP = 2


def cuda_graphs_enabled(*, explicit: bool | None = None, deterministic: bool = False) -> bool:
    """Whether CUDA graph training is allowed for this run."""
    if deterministic:
        return False
    if explicit is False:
        return False
    raw = os.environ.get("SEISO_CUDA_GRAPHS", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    if explicit is True:
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
        if major < 8:
            return False
    except ImportError:
        return False

    try:
        from seiso.kernels.arch_tuning import detect_arch_tuning

        return detect_arch_tuning().use_cuda_graphs
    except ImportError:
        return True


class CudaGraphTrainingManager:
    """
    Capture forward + backward on fixed tensor addresses and replay each step.

    Graph contains: compute_loss → loss scaling → accelerator.backward.
    Optimizer step stays outside the graph (HF Trainer handles grad accumulation).
    Falls back to eager when capture fails or batch shapes change.
    """

    def __init__(self, *, warmup_steps: int = _WARMUP_STEPS) -> None:
        self._warmup_steps = warmup_steps
        self._graph: Any | None = None
        self._static_inputs: dict[str, Any] | None = None
        self._static_loss: Any | None = None
        self._captured = False
        self._enabled = False
        self._eager_steps = 0
        self._capture_failed = False
        self._last_shape_key: tuple | None = None

    @property
    def active(self) -> bool:
        return self._enabled and self._captured and not self._capture_failed

    def metadata(self) -> dict[str, Any]:
        return {
            "cuda_graphs_enabled": self._enabled,
            "cuda_graphs_captured": self._captured,
            "cuda_graphs_active": self.active,
            "cuda_graphs_eager_steps": self._eager_steps,
        }

    def try_enable(self, *, explicit: bool | None = None, deterministic: bool = False) -> bool:
        self._enabled = cuda_graphs_enabled(explicit=explicit, deterministic=deterministic)
        return self._enabled

    def reset(self) -> None:
        self._graph = None
        self._static_inputs = None
        self._static_loss = None
        self._captured = False
        self._eager_steps = 0
        self._capture_failed = False
        self._last_shape_key = None

    @staticmethod
    def _shape_key(inputs: dict[str, Any]) -> tuple:
        parts = []
        for key in sorted(inputs.keys()):
            val = inputs[key]
            if hasattr(val, "shape") and hasattr(val, "dtype"):
                parts.append((key, tuple(val.shape), str(val.dtype), val.device.type))
        return tuple(parts)

    @staticmethod
    def _clone_static_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        static: dict[str, Any] = {}
        for key, val in inputs.items():
            if hasattr(val, "clone") and hasattr(val, "is_cuda") and val.is_cuda:
                static[key] = val.clone()
            else:
                static[key] = val
        return static

    @staticmethod
    def _copy_inputs(src: dict[str, Any], dst: dict[str, Any]) -> None:
        for key, val in src.items():
            static = dst.get(key)
            if static is not None and hasattr(static, "copy_") and hasattr(val, "shape"):
                static.copy_(val)

    @staticmethod
    def _model_uses_bnb_quant(model: Any) -> bool:
        try:
            for module in model.modules():
                cls = type(module).__name__.lower()
                if "bnb" in cls or "bitsandbytes" in cls:
                    return True
                if hasattr(module, "weight") and hasattr(module.weight, "quant_state"):
                    return True
        except ImportError:
            pass
        return False

    def eligible(self, trainer: Any, model: Any | None = None) -> bool:
        if not self._enabled or self._capture_failed:
            return False
        try:
            import torch
        except ImportError:
            return False
        if not torch.cuda.is_available():
            return False

        args = trainer.args
        if getattr(args, "gradient_checkpointing", False):
            return False
        if getattr(args, "fp16", False):
            return False
        if getattr(args, "torch_compile", False):
            return False
        if int(getattr(args, "n_gpu", 1)) > 1:
            return False
        if getattr(args, "deepspeed", None):
            return False
        if getattr(args, "use_cpu", False):
            return False
        return not (model is not None and self._model_uses_bnb_quant(model))

    def _compute_step_loss(
        self,
        trainer: Any,
        model: Any,
        inputs: dict[str, Any],
        num_items_in_batch: Any,
    ) -> Any:

        with trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if int(getattr(trainer.args, "n_gpu", 1)) > 1:
            loss = loss.mean()

        if (
            not getattr(trainer, "model_accepts_loss_kwargs", False) or num_items_in_batch is None
        ) and getattr(trainer, "compute_loss_func", None) is None:
            loss = loss / trainer.current_gradient_accumulation_steps

        return loss

    def _backward(self, trainer: Any, loss: Any) -> None:
        kwargs: dict[str, Any] = {}
        if getattr(trainer.args, "optim", "") in ("lomo", "adalomo"):
            kwargs["learning_rate"] = trainer._get_learning_rate()
        trainer.accelerator.backward(loss, **kwargs)

    def _try_capture(
        self,
        trainer: Any,
        model: Any,
        inputs: dict[str, Any],
        num_items_in_batch: Any,
    ) -> bool:
        import torch

        try:
            self._static_inputs = self._clone_static_inputs(inputs)
            self._last_shape_key = self._shape_key(inputs)

            for _ in range(max(_CAPTURE_WARMUP - 1, 0)):
                with torch.no_grad():
                    self._compute_step_loss(trainer, model, self._static_inputs, num_items_in_batch)
            warmup_loss = self._compute_step_loss(
                trainer, model, self._static_inputs, num_items_in_batch
            )
            self._backward(trainer, warmup_loss)
            torch.cuda.synchronize()
            trainer.optimizer.zero_grad(set_to_none=True)

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._static_loss = self._compute_step_loss(
                    trainer, model, self._static_inputs, num_items_in_batch
                )
                self._backward(trainer, self._static_loss)

            self._captured = True
            logger.info(
                "CUDA training graph captured (forward+backward, shape=%s)",
                self._last_shape_key,
            )
            return True
        except Exception as exc:
            logger.warning("CUDA graph capture failed — using eager training: %s", exc)
            self._capture_failed = True
            self._graph = None
            self._static_inputs = None
            self._static_loss = None
            self._captured = False
            return False

    def training_step(
        self,
        trainer: Any,
        model: Any,
        inputs: dict[str, Any],
        num_items_in_batch: Any = None,
    ) -> Any | None:
        """
        Run a CUDA-graph training step. Returns loss tensor, or None to fall back to eager.
        """
        if not self._enabled or self._capture_failed:
            return None
        if self._model_uses_bnb_quant(model):
            if not getattr(self, "_bnb_skip_logged", False):
                logger.info(
                    "CUDA graphs skipped — bitsandbytes quantized layers are not graph-safe"
                )
                self._bnb_skip_logged = True
            return None
        if not self.eligible(trainer, model):
            return None

        model.train()
        if hasattr(trainer.optimizer, "train") and callable(trainer.optimizer.train):
            trainer.optimizer.train()

        inputs = trainer._prepare_inputs(inputs)
        shape_key = self._shape_key(inputs)
        if self._captured and shape_key != self._last_shape_key:
            logger.info("CUDA graph reset — batch shape changed")
            self.reset()
            self._enabled = True

        if not self._captured:
            if self._eager_steps < self._warmup_steps:
                self._eager_steps += 1
                loss = self._compute_step_loss(trainer, model, inputs, num_items_in_batch)
                self._backward(trainer, loss)
                return loss.detach()

            if not self._try_capture(trainer, model, inputs, num_items_in_batch):
                loss = self._compute_step_loss(trainer, model, inputs, num_items_in_batch)
                self._backward(trainer, loss)
                return loss.detach()

        if self._static_inputs is None or self._graph is None:
            return None

        self._copy_inputs(inputs, self._static_inputs)
        self._graph.replay()
        return self._static_loss.detach()


class CudaGraphTrainerMixin:
    """Mixin for HF/TRL trainers — overrides training_step with CUDA graph replay."""

    _seiso_cuda_graph_manager: CudaGraphTrainingManager | None = None

    def __init__(self, *args, use_cuda_graphs: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._seiso_cuda_graph_manager = CudaGraphTrainingManager()
        deterministic = bool(getattr(getattr(self, "args", None), "full_determinism", False))
        self._seiso_cuda_graph_manager.try_enable(
            explicit=use_cuda_graphs,
            deterministic=deterministic,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        mgr = self._seiso_cuda_graph_manager
        if mgr is not None:
            loss = mgr.training_step(self, model, inputs, num_items_in_batch)
            if loss is not None:
                return loss
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mgr = self._seiso_cuda_graph_manager
        if mgr is not None and mgr.active:
            logs = dict(logs)
            logs.update(mgr.metadata())
        return super().log(logs, start_time=start_time)


def attach_cuda_graphs(trainer: Any, *, enabled: bool = True, deterministic: bool = False) -> Any:
    """Attach CUDA graph manager to an existing trainer instance (monkeypatch)."""
    mgr = CudaGraphTrainingManager()
    mgr.try_enable(explicit=enabled, deterministic=deterministic)
    trainer._seiso_cuda_graph_manager = mgr

    if getattr(trainer, "_seiso_cuda_graph_patched", False):
        return trainer

    orig_step = trainer.training_step

    def _patched_training_step(model, inputs, num_items_in_batch=None):
        loss = mgr.training_step(trainer, model, inputs, num_items_in_batch)
        if loss is not None:
            return loss
        return orig_step(model, inputs, num_items_in_batch=num_items_in_batch)

    trainer.training_step = _patched_training_step
    trainer._seiso_cuda_graph_patched = True
    return trainer


def make_training_graph_callback(
    *, deterministic: bool = False, enabled: bool = True
) -> Any | None:
    """TrainerCallback that resets CUDA graph state after training."""
    try:
        from transformers import TrainerCallback
    except ImportError:
        return None

    class _CudaGraphCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            t = kwargs.get("trainer")
            if t is not None and getattr(t, "_seiso_cuda_graph_manager", None) is None:
                attach_cuda_graphs(t, enabled=enabled, deterministic=deterministic)

        def on_train_end(self, args, state, control, **kwargs):
            t = kwargs.get("trainer")
            if t is not None and getattr(t, "_seiso_cuda_graph_manager", None) is not None:
                t._seiso_cuda_graph_manager.reset()

    return _CudaGraphCallback()
