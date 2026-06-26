"""Smart Router serve command."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer

from seiso_cli.console import console


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "seiso_cli").is_dir():
            return parent
    return Path.cwd()


def _start_router_compose_stack(*, vllm: bool, detach: bool) -> None:
    root = _resolve_repo_root()
    router_dir = root / "deploy" / "model-router"
    compose_file = (
        "docker-compose.local.vllm.yml" if vllm else "docker-compose.local.yml"
    )
    compose_path = router_dir / compose_file
    if not compose_path.is_file():
        raise typer.BadParameter(f"missing compose file: {compose_path}")

    cmd = ["docker", "compose", "-f", compose_file, "up", "--build"]
    if detach:
        cmd.append("-d")
    stack_label = "vLLM + LiteLLM" if vllm else "llama.cpp"
    console.print(f"[bold green]Seiso Router stack[/] ({stack_label}) via {compose_path}")
    if vllm:
        console.print("Router: http://127.0.0.1:8780 · Enable Forge: SEISO_MODEL_ROUTER_ENABLED=true")
    subprocess.run(cmd, cwd=router_dir, check=True, env=os.environ.copy())


def router_serve(
    config: str = typer.Option(
        "deploy/model-router/config/router.local.yaml",
        "--config",
        "-c",
        help="Router YAML config",
    ),
    vllm: bool = typer.Option(
        False,
        "--vllm",
        help="Use deploy/model-router/config/router.local.vllm.yaml (LiteLLM + vLLM stack)",
    ),
    stack: str | None = typer.Option(
        None,
        "--stack",
        help="Start Docker stack instead of uvicorn: llamacpp | vllm",
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        "-d",
        help="With --stack, run docker compose detached",
    ),
    host: str | None = typer.Option(None, help="Bind host"),
    port: int | None = typer.Option(None, help="Bind port"),
    reload: bool = typer.Option(False, help="Dev auto-reload"),
) -> None:
    """Launch Seiso Smart Router (llama.cpp or vLLM specialists via LiteLLM)."""
    stack_norm = (stack or "").strip().lower()
    if stack_norm in {"vllm", "llamacpp", "llama.cpp", "llama_cpp"}:
        _start_router_compose_stack(vllm=stack_norm == "vllm", detach=detach)
        return
    if stack_norm:
        raise typer.BadParameter("--stack must be llamacpp or vllm")

    import uvicorn

    from seiso.model_router.config import RouterSettings, resolve_paths

    config_path = Path(config)
    if vllm:
        config_path = Path("deploy/model-router/config/router.local.vllm.yaml")
    settings = resolve_paths(RouterSettings.load(config_path), base=_resolve_repo_root())
    bind_host = host or settings.host
    bind_port = port or settings.port
    backend = settings.inference_backend
    console.print(
        f"[bold green]Seiso Router[/] ({backend}) → http://{bind_host}:{bind_port}"
    )
    if settings.litellm_gateway_enabled():
        console.print("Completions dispatch via LiteLLM (vLLM stack must be running)")
    uvicorn.run(
        "seiso.model_router.main:build_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level="info",
    )