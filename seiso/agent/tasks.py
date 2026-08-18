"""Task / plan types for the agentic OS harness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from seiso.compat import StrEnum


class TaskKind(StrEnum):
    """What a harness step is trying to do."""

    CHAT = "chat"
    CODE = "code"
    EMBED = "embed"
    DRAFT = "draft"
    TARGET = "target"
    INFERENCE = "inference"
    FINETUNE = "finetune"
    SLIME = "slime"
    DISTILL_RL = "distill_rl"
    NEMO_RL = "nemo_rl"
    EXPORT = "export"
    COMPRESS = "compress"
    DOCTOR = "doctor"


# Job kinds the marketplace / compute kernel understands.
JOB_KINDS = frozenset(
    {
        TaskKind.INFERENCE.value,
        TaskKind.FINETUNE.value,
        TaskKind.SLIME.value,
        TaskKind.DISTILL_RL.value,
        TaskKind.NEMO_RL.value,
        TaskKind.CHAT.value,
        TaskKind.CODE.value,
        TaskKind.EXPORT.value,
        TaskKind.COMPRESS.value,
        TaskKind.DOCTOR.value,
    }
)


def parse_task_kind(raw: str | TaskKind) -> TaskKind:
    if isinstance(raw, TaskKind):
        return raw
    text = str(raw).strip().lower().replace("-", "_")
    try:
        return TaskKind(text)
    except ValueError as exc:
        allowed = ", ".join(k.value for k in TaskKind)
        raise ValueError(f"unknown task kind {raw!r}; expected one of: {allowed}") from exc


def job_kind_for_task(kind: TaskKind | str) -> str:
    """Map a task to the marketplace / compute job kind."""
    parsed = parse_task_kind(kind)
    if parsed in {TaskKind.CHAT, TaskKind.CODE, TaskKind.EMBED, TaskKind.DRAFT, TaskKind.TARGET}:
        return TaskKind.INFERENCE.value
    return parsed.value


@dataclass(frozen=True, slots=True)
class Step:
    """One harness step."""

    id: str
    kind: TaskKind
    action: str = "run"
    payload: Mapping[str, Any] = field(default_factory=dict)
    required_context: int = 2048
    estimated_vram_mb: int = 0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["payload"] = dict(self.payload)
        return data


@dataclass(frozen=True, slots=True)
class Plan:
    """Ordered harness plan. Executors never see secrets from the caller."""

    id: str
    goal: str
    steps: tuple[Step, ...]
    route_class: str = "allow_paid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "route_class": self.route_class,
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class StepResult:
    step_id: str
    status: str
    compute_target: str | None = None
    route_backend: str | None = None
    route_model: str | None = None
    detail: str = ""
    output: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output"] = dict(self.output)
        return data
