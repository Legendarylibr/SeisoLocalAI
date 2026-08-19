"""Task kind → preferred local candidate roles."""

from __future__ import annotations

from seiso.agent.tasks import TaskKind

# First matching downloaded candidate role wins preference order.
ROLES_FOR_TASK: dict[TaskKind, tuple[str, ...]] = {
    TaskKind.CHAT: ("chat",),
    TaskKind.CODE: ("code", "chat"),
    TaskKind.EMBED: ("embed",),
    TaskKind.DRAFT: ("draft",),
    TaskKind.TARGET: ("target",),
    TaskKind.INFERENCE: ("chat", "code"),
    TaskKind.FINETUNE: ("chat",),
    TaskKind.SLIME: ("chat",),
    TaskKind.DISTILL_RL: ("chat",),
    TaskKind.NEMO_RL: ("chat",),
    TaskKind.EXPORT: ("chat",),
    TaskKind.COMPRESS: ("chat",),
    TaskKind.DOCTOR: ("chat",),
}


def roles_for_task(task: TaskKind) -> tuple[str, ...]:
    return ROLES_FOR_TASK.get(task, ("chat",))
