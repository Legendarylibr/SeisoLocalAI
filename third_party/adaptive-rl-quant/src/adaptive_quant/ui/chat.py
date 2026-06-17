"""Interactive chat and continuous-learning bridge for the launcher dashboard."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import validate_online_prompt_text
from adaptive_quant.logging_utils import write_json
from adaptive_quant.reward_path import RewardPathTracker
from adaptive_quant.rl_loop import run_rl_episode
from adaptive_quant.trainer import Trainer, build_trainer
from adaptive_quant.types import HardwareType, PromptSample
from adaptive_quant.ui.model_selection import (
    discover_llama_cpp_binary,
    download_route_model,
    load_selected_model_id,
    model_catalog_payload,
    resolve_model_choice,
    save_selected_model_id,
    search_huggingface_gguf_repos,
)

CHAT_TASKS_REL_PATH = "outputs/chat_tasks.jsonl"
SESSION_DIR_REL = "outputs/.launcher_chat_session"
SESSION_CONFIG_NAME = "session_config.json"
SESSION_CHECKPOINT_NAME = "policy_checkpoint.json"
SESSION_REWARD_PATH_NAME = "reward_path_state.json"
_RL_BACKENDS = frozenset({"simulator_rl", "continuous_learn", "llama_cpp_rl"})


def default_chat_tasks_path(repo: Path) -> Path:
    return repo / CHAT_TASKS_REL_PATH


def _read_session_meta(repo: Path) -> dict[str, Any]:
    config_path = repo / SESSION_DIR_REL / SESSION_CONFIG_NAME
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "selected_model_id": payload.get("selected_model_id"),
        "measurement_backend": payload.get("backend"),
    }


def build_chat_config(*, repo: Path) -> dict[str, Any]:
    catalog = model_catalog_payload(repo=repo)
    llama_ready = bool(catalog.get("llama_ready"))

    tasks_path = default_chat_tasks_path(repo)
    task_count = 0
    if tasks_path.is_file():
        task_count = sum(
            1 for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    backends: list[dict[str, Any]] = [
        {
            "id": "simulator_rl",
            "label": "Simulator RL (preview)",
            "description": "Run one RL episode against the built-in simulator.",
        },
        {
            "id": "continuous_learn",
            "label": "Continuous learning (simulator)",
            "description": "Apply a policy update using simulator measurements.",
        },
    ]
    if llama_ready:
        backends.extend(
            [
                {
                    "id": "llama_cpp_rl",
                    "label": "llama.cpp RL (train)",
                    "description": "RL episode measured by your local llama.cpp binary + GGUF.",
                    "available": True,
                },
                {
                    "id": "llama_cpp",
                    "label": "llama.cpp completion",
                    "description": "Generate text with the selected Hugging Face / local model.",
                    "available": True,
                },
            ]
        )
    else:
        backends.append(
            {
                "id": "llama_cpp",
                "label": "llama.cpp completion",
                "description": "Download a Hugging Face GGUF route or configure llama.cpp paths.",
                "available": False,
            }
        )

    session = _read_session_meta(repo)
    return {
        "backends": backends,
        "default_backend": "simulator_rl",
        "chat_tasks_path": str(tasks_path.relative_to(repo)),
        "chat_tasks_count": task_count,
        "session_dir": SESSION_DIR_REL,
        **catalog,
        "measurement_backend": session.get("measurement_backend", "simulator"),
    }


class ChatSessionManager:
    def __init__(self, *, repo: Path) -> None:
        self.repo = repo
        self._lock = threading.Lock()
        self._session_dir = repo / SESSION_DIR_REL
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._trainer: Trainer | None = None
        self._trainer_fingerprint: tuple[Any, ...] | None = None
        self._reward_tracker = RewardPathTracker()
        self._task_counter = 0
        self._load_session_state()

    def _load_session_state(self) -> None:
        reward_path = self._session_dir / SESSION_REWARD_PATH_NAME
        if reward_path.is_file():
            self._reward_tracker = RewardPathTracker.load(reward_path)

    def _default_session_config(self) -> FrameworkConfig:
        return FrameworkConfig(
            run_name="launcher_chat_session",
            outputs_dir=str(self.repo / "outputs"),
            backend="simulator",
            stability_probe_count=1,
            training_episodes=1,
            evaluation_episodes=1,
            seed=17,
        )

    def _session_config(self) -> FrameworkConfig:
        config_path = self._session_dir / SESSION_CONFIG_NAME
        if config_path.is_file():
            return FrameworkConfig.from_file(str(config_path))
        return self._default_session_config()

    @staticmethod
    def _config_fingerprint(config: FrameworkConfig) -> tuple[Any, ...]:
        return (
            config.backend,
            config.llama_cpp_binary,
            config.llama_cpp_model,
            config.training_backend,
        )

    def _persist_session_config(self, config: FrameworkConfig, *, model_id: str | None) -> None:
        payload = config.to_flat_dict()
        if model_id:
            payload["selected_model_id"] = model_id
        write_json(str(self._session_dir / SESSION_CONFIG_NAME), payload)
        save_selected_model_id(self.repo, model_id)

    def _resolve_training_config(
        self,
        body: dict[str, Any],
        *,
        backend: str,
    ) -> tuple[FrameworkConfig, str | None]:
        config = self._session_config()
        model_id_raw = body.get("model_id")
        model_id = str(model_id_raw).strip() if model_id_raw is not None else None
        if model_id == "":
            model_id = None
        if model_id is None:
            model_id = load_selected_model_id(self.repo)

        use_llama = backend in {"llama_cpp", "llama_cpp_rl"} or (
            backend == "continuous_learn" and model_id is not None
        )
        if use_llama:
            choice = resolve_model_choice(repo=self.repo, model_id=model_id)
            if choice is None or not choice.ready or not choice.model_path:
                raise ValueError(
                    "Select a downloaded Hugging Face route or local GGUF model first."
                )
            binary = discover_llama_cpp_binary(repo=self.repo) or config.llama_cpp_binary
            if not binary:
                raise FileNotFoundError(
                    "llama.cpp binary not found. Set LLAMA_CPP_BINARY, llama_cpp_binary in config, "
                    "or install llama-cli on PATH."
                )
            config = config.clone(
                backend="llama_cpp",
                llama_cpp_binary=binary,
                llama_cpp_model=choice.model_path,
            )
            model_id = choice.id
        elif backend == "continuous_learn":
            config = config.clone(backend="simulator")
            model_id = None
        else:
            config = config.clone(backend="simulator")
            if model_id and backend == "simulator_rl":
                save_selected_model_id(self.repo, model_id)
            model_id = model_id if backend == "simulator_rl" else None

        if use_llama or backend in {"continuous_learn", "llama_cpp_rl", "llama_cpp"}:
            self._persist_session_config(config, model_id=model_id)
        return config, model_id

    def _ensure_trainer(self, config: FrameworkConfig) -> Trainer:
        fingerprint = self._config_fingerprint(config)
        if self._trainer is not None and self._trainer_fingerprint == fingerprint:
            return self._trainer

        if self._trainer is not None:
            self._trainer.close()
            self._trainer = None

        log_path = str(self._session_dir / "episodes.jsonl")
        trainer = build_trainer(config, log_path=log_path)
        checkpoint = self._session_dir / SESSION_CHECKPOINT_NAME
        if checkpoint.is_file():
            trainer.load_checkpoint(str(checkpoint))
        self._trainer = trainer
        self._trainer_fingerprint = fingerprint
        return trainer

    def _save_session(self) -> None:
        if self._trainer is not None:
            self._trainer.save_checkpoint(str(self._session_dir / SESSION_CHECKPOINT_NAME))
        self._reward_tracker.save(self._session_dir / SESSION_REWARD_PATH_NAME)

    def handle_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        message = validate_online_prompt_text(str(body.get("message") or ""))
        if not message.strip():
            raise ValueError("message is required")

        backend = str(body.get("backend") or "simulator_rl").strip().lower()
        hardware_raw = str(body.get("hardware") or "gpu").strip().lower()
        hardware = HardwareType(hardware_raw)
        learn = backend in {"continuous_learn", "llama_cpp_rl"}
        append_task = bool(body.get("append_task", True))

        if backend == "llama_cpp":
            config, model_id = self._resolve_training_config(body, backend=backend)
            return self._handle_llama_cpp(message, config=config, model_id=model_id)

        if backend not in _RL_BACKENDS:
            raise ValueError(f"unsupported chat backend: {backend!r}")

        with self._lock:
            config, model_id = self._resolve_training_config(body, backend=backend)
            trainer = self._ensure_trainer(config)
            self._task_counter += 1
            prompt_id = str(body.get("prompt_id") or f"chat_{self._task_counter:06d}")
            domain = str(body.get("prompt_domain") or "chat")
            prompt = PromptSample(prompt_id=prompt_id, text=message, domain=domain)

            outcome = run_rl_episode(
                trainer,
                episode_index=self._task_counter,
                phase="train",
                forced_prompt=prompt,
                forced_hardware=hardware,
                deterministic=not learn,
                apply_update=learn,
                log_episode=True,
            )
            self._reward_tracker.record(prompt_id, outcome.reward, domain=domain)
            if learn:
                self._save_session()

            task_record = None
            if append_task:
                task_record = append_chat_task(
                    self.repo,
                    prompt_id=prompt_id,
                    prompt_text=message,
                    prompt_domain=domain,
                    hardware=hardware.value,
                    reward=outcome.reward,
                )

            metrics = outcome.result.metrics
            measurement = config.backend
            return {
                "backend": backend,
                "measurement_backend": measurement,
                "selected_model_id": model_id,
                "response_text": _format_rl_response(outcome, learn=learn, measurement=measurement),
                "prompt_id": prompt_id,
                "reward": outcome.reward,
                "learn_applied": learn,
                "metrics": {
                    "latency_ms": float(getattr(metrics, "latency_ms", 0.0)),
                    "throughput_tps": float(getattr(metrics, "throughput_tps", 0.0)),
                    "memory_mb": float(getattr(metrics, "memory_mb", 0.0)),
                    "perplexity": float(getattr(metrics, "perplexity", 0.0)),
                },
                "decision": {
                    "mode": outcome.decision.mode.value,
                    "bits": outcome.decision.base_bit_width,
                    "scale": outcome.decision.scale_factor,
                    "clip": outcome.decision.clipping_range,
                },
                "reward_path_tail": self._reward_tracker.reward_path_tail(limit=8),
                "chat_task": task_record,
            }

    def _handle_llama_cpp(
        self,
        message: str,
        *,
        config: FrameworkConfig,
        model_id: str | None,
    ) -> dict[str, Any]:
        from adaptive_quant.backends.llama_cpp import (
            require_llama_cpp_paths,
            run_llama_cpp_completion,
        )

        llama_cpp_binary, llama_cpp_model = require_llama_cpp_paths(config)
        started = time.perf_counter()
        metrics, generated = run_llama_cpp_completion(
            config,
            llama_cpp_binary=llama_cpp_binary,
            llama_cpp_model=llama_cpp_model,
            prompt_text=message,
            ngl=999,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "backend": "llama_cpp",
            "measurement_backend": "llama_cpp",
            "selected_model_id": model_id,
            "response_text": generated,
            "reward": None,
            "learn_applied": False,
            "metrics": {
                "latency_ms": float(metrics.get("latency_ms", latency_ms)),
                "throughput_tps": float(metrics.get("throughput_tps", 0.0)),
                "memory_mb": float(metrics.get("memory_mb", 0.0)),
                "perplexity": float(metrics.get("perplexity", 0.0)),
            },
        }

    def list_tasks(self) -> dict[str, Any]:
        path = default_chat_tasks_path(self.repo)
        if not path.is_file():
            return {"path": str(path.relative_to(self.repo)), "tasks": [], "count": 0}
        tasks: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                tasks.append(payload)
        return {"path": str(path.relative_to(self.repo)), "tasks": tasks[-50:], "count": len(tasks)}

    def reset_session(self) -> dict[str, str]:
        with self._lock:
            if self._trainer is not None:
                self._trainer.close()
                self._trainer = None
                self._trainer_fingerprint = None
            for name in (
                SESSION_CONFIG_NAME,
                SESSION_CHECKPOINT_NAME,
                SESSION_REWARD_PATH_NAME,
                "episodes.jsonl",
            ):
                target = self._session_dir / name
                if target.is_file():
                    target.unlink()
            self._reward_tracker = RewardPathTracker()
            self._task_counter = 0
            save_selected_model_id(self.repo, None)
            return {"status": "session_reset", "session_dir": SESSION_DIR_REL}


def _format_rl_response(outcome: Any, *, learn: bool, measurement: str) -> str:
    decision = outcome.decision
    metrics = outcome.result.metrics
    mode = "learned" if learn else "preview"
    via = "llama.cpp" if measurement == "llama_cpp" else "simulator"
    return (
        f"[{mode} via {via}] Policy -> {decision.mode.value} @ {decision.base_bit_width}-bit "
        f"(scale={decision.scale_factor:.3f}, clip={decision.clipping_range:.3f}). "
        f"Reward={outcome.reward:.4f}, latency={metrics.latency_ms:.2f}ms, "
        f"throughput={metrics.throughput_tps:.1f} tps, perplexity={metrics.perplexity:.3f}."
    )


def append_chat_task(
    repo: Path,
    *,
    prompt_id: str,
    prompt_text: str,
    prompt_domain: str = "chat",
    hardware: str = "gpu",
    reward: float | None = None,
) -> dict[str, Any]:
    path = default_chat_tasks_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "prompt_domain": prompt_domain,
        "hardware": hardware,
    }
    if reward is not None:
        record["last_reward"] = float(reward)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"path": str(path.relative_to(repo)), "prompt_id": prompt_id}


def build_models_response(*, repo: Path, body: dict[str, Any] | None = None) -> dict[str, Any]:
    action = str((body or {}).get("action") or "catalog").strip().lower()
    if action == "search":
        query = str((body or {}).get("query") or "").strip()
        return {
            "query": query,
            "results": search_huggingface_gguf_repos(query=query),
        }
    if action == "select":
        model_id = str((body or {}).get("model_id") or "").strip() or None
        save_selected_model_id(repo, model_id)
        payload = model_catalog_payload(repo=repo)
        payload["selected_model_id"] = model_id
        return payload
    if action == "download":
        model_id = str((body or {}).get("model_id") or "").strip()
        choice = resolve_model_choice(repo=repo, model_id=model_id)
        if choice is None or not choice.route_id:
            raise ValueError("model_id must reference a Hugging Face route catalog entry")
        downloaded = download_route_model(repo=repo, route_id=choice.route_id)
        save_selected_model_id(repo, downloaded.id)
        payload = model_catalog_payload(repo=repo)
        payload["downloaded"] = downloaded.to_dict()
        return payload
    return model_catalog_payload(repo=repo)


def build_chat_response(
    *, repo: Path, body: dict[str, Any], session: ChatSessionManager
) -> dict[str, Any]:
    action = str(body.get("action") or "message").strip().lower()
    if action == "list_tasks":
        return session.list_tasks()
    if action == "reset_session":
        return session.reset_session()
    return session.handle_chat(body)
