"""Frontend vs agent training policy.

Frontend (Forge UI / ``POST /api/training/jobs``) exposes the full local
training config (method, quant, local multi-GPU DDP, hyperparams) but refuses
multi-node / mesh. Mesh and ``distributed_num_nodes > 1`` are agent-only and
require a Buzz agent identity.
"""

from __future__ import annotations

from typing import Any

from seiso.agent.surface import TrainingSurface, buzz_agent_present

# Forge UI / browser sessions always run as this surface.
FRONTEND_SURFACE = TrainingSurface.FRONTEND
FRONTEND_MAX_NODES = 1

# Stable keys the frontend should keep exposing (full local training config).
FRONTEND_TRAINING_CONFIG_FIELDS: tuple[str, ...] = (
    "model_id",
    "dataset",
    "dataset_format",
    "method",
    "quant",
    "epochs",
    "batch_size",
    "learning_rate",
    "max_seq_length",
    "gradient_accumulation_steps",
    "logging_steps",
    "save_steps",
    "max_eval_samples",
    "lora_r",
    "lora_alpha",
    "multi_gpu",
    "distributed_strategy",
    "distributed_nproc_per_node",
    "distributed_num_nodes",
    "distributed_node_rank",
    "distributed_master_addr",
    "distributed_master_port",
    "ddp_backend",
    "ddp_find_unused_parameters",
    "use_triton",
    "use_fused_ce",
    "use_fused_lora",
    "packing",
    "train_on_responses_only",
)


def assert_surface_distributed_config(
    surface: TrainingSurface | str,
    config: dict[str, Any] | Any,
) -> None:
    """Refuse multi-node on the frontend surface; agents need Buzz+mesh."""
    surface_val = (
        surface
        if isinstance(surface, TrainingSurface)
        else TrainingSurface(str(surface).strip().lower())
    )
    if hasattr(config, "model_dump"):
        cfg = config.model_dump()
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = {
            "distributed_num_nodes": getattr(config, "distributed_num_nodes", 1),
            "nemo_rl_num_nodes": getattr(config, "nemo_rl_num_nodes", 1),
        }
    # Cover Accelerate + NeMo RL cluster node counts (UI pins both to 1).
    nnodes = max(
        int(cfg.get("distributed_num_nodes", 1) or 1),
        int(cfg.get("nemo_rl_num_nodes", 1) or 1),
    )
    if surface_val == TrainingSurface.FRONTEND and nnodes > FRONTEND_MAX_NODES:
        raise ValueError(
            "Multi-node training (distributed_num_nodes>1 or nemo_rl_num_nodes>1) "
            "is Buzz-agent/mesh-only. The Forge UI exposes full local training "
            "config with nodes=1 (including local multi-GPU DDP). Use a Buzz "
            "agent + `seiso mesh` for multi-node coordination."
        )
    if nnodes > FRONTEND_MAX_NODES:
        require_multinode_mesh_agent(nnodes)


def require_multinode_mesh_agent(nnodes: int) -> None:
    """Fail closed: multi-node only under Buzz agent nsec + SEISO_ALLOW_MESH."""
    if nnodes <= FRONTEND_MAX_NODES:
        return
    from seiso.agent.nostr_identity import require_buzz_nsec
    from seiso.mesh.flags import mesh_allowed

    if not mesh_allowed():
        raise ValueError(
            "Multi-node training (distributed_num_nodes>1) is Buzz-agent/mesh-only. "
            "Set SEISO_ALLOW_MESH=1 and BUZZ_PRIVATE_KEY (valid nsec), "
            "or keep distributed_num_nodes=1 for local multi-GPU. "
            "Not functional for real multi-node yet."
        )
    try:
        require_buzz_nsec(feature="Multi-node training")
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def frontend_training_surface() -> dict[str, Any]:
    """Capabilities + config cover for Forge UI — full local training, no mesh."""
    from seiso.mesh.flags import mesh_allowed

    return {
        "surface": TrainingSurface.FRONTEND.value,
        "exposes_full_training_config": True,
        "config_fields": list(FRONTEND_TRAINING_CONFIG_FIELDS),
        "local_distributed": {
            "enabled": True,
            "max_nodes": FRONTEND_MAX_NODES,
            "strategies": ["auto", "none", "ddp"],
            "multi_gpu": True,
            "note": (
                "Local multi-GPU DDP (nnodes=1) is fully supported. "
                "Multi-node / mesh is Buzz-agent-only."
            ),
        },
        "multi_node": False,
        "mesh": {
            "available_on_this_surface": False,
            "opt_in_env": "SEISO_ALLOW_MESH",
            "requires": "buzz_agent_nsec",
            "operator_mesh_flag_set": mesh_allowed(),
            "buzz_agent_present": buzz_agent_present(),
            "nostr_signed_plans": True,
            "sig_alg": "bip340-schnorr",
        },
        "agent_surface": TrainingSurface.AGENT.value,
        "buzz_compatible": True,
    }


def agent_training_surface() -> dict[str, Any]:
    """Capabilities for generic agents (Buzz-compatible)."""
    from seiso.agent.nostr_identity import get_buzz_keypair
    from seiso.mesh.flags import mesh_allowed

    mesh_ok = bool(mesh_allowed() and get_buzz_keypair())
    return {
        "surface": TrainingSurface.AGENT.value,
        "exposes_full_training_config": True,
        "config_fields": list(FRONTEND_TRAINING_CONFIG_FIELDS),
        "local_distributed": {
            "enabled": True,
            "max_nodes": None,
            "strategies": ["auto", "none", "ddp"],
            "multi_gpu": True,
        },
        "multi_node": mesh_ok,
        "mesh": {
            "available_on_this_surface": mesh_ok,
            "opt_in_env": "SEISO_ALLOW_MESH",
            "requires": "buzz_agent_nsec",
            "operator_mesh_flag_set": mesh_allowed(),
            "buzz_agent_present": buzz_agent_present(),
            "nostr_signed_plans": True,
            "sig_alg": "bip340-schnorr",
            "not_functional_yet": True,
        },
        "buzz_compatible": True,
    }
