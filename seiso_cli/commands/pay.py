"""CLI for the experimental opt-in Seiso sats marketplace (remote only)."""

from __future__ import annotations

import json
import os
from typing import Annotated, Any

import typer

from seiso_cli.console import console

pay_app = typer.Typer(
    name="pay",
    help=(
        "Experimental sats marketplace client/operator tools "
        "(opt-in: SEISO_ALLOW_PAY=1; not functional for real funds yet)."
    ),
    no_args_is_help=True,
)


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str))


def _require_allow() -> None:
    from seiso.pay.flags import require_pay_allowed

    try:
        require_pay_allowed()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


@pay_app.command("quote")
def pay_quote(
    type_: Annotated[
        str,
        typer.Option("--type", help="finetune|slime|distill_rl|rl_quant|nemo_rl|inference"),
    ],
    preset: Annotated[str | None, typer.Option(help="smoke|minimal|…")] = None,
    prompt_tokens: Annotated[int, typer.Option(help="For inference quotes")] = 0,
    completion_tokens: Annotated[int, typer.Option(help="For inference quotes")] = 0,
    flat_call: Annotated[bool, typer.Option(help="Flat sats/call inference")] = False,
) -> None:
    """Show compute + protocol fee split (no charge). Requires SEISO_ALLOW_PAY=1."""
    _require_allow()
    from seiso.pay.pricing import quote_inference_tokens, quote_job

    if type_.strip().lower() == "inference":
        _print_json(
            quote_inference_tokens(
                prompt_tokens, completion_tokens, flat_call=flat_call
            )
        )
        return
    try:
        _print_json(quote_job(type_, preset=preset))
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc


@pay_app.command("session")
def pay_session(
    action: Annotated[
        str, typer.Argument(help="create|status|fund")
    ],
    scopes: Annotated[
        str, typer.Option(help="Comma scopes: inference,finetune,rl")
    ] = "inference,finetune,rl",
    sats: Annotated[int, typer.Option(help="Fund amount (create/fund)")] = 0,
    session: Annotated[
        str | None, typer.Option("--session", help="Session id")
    ] = None,
    token: Annotated[
        str | None, typer.Option(help="Pay token (status via token)")
    ] = None,
    faucet: Annotated[
        bool, typer.Option(help="Use dev faucet (requires SEISO_PAY_FAUCET=1)")
    ] = False,
    l402: Annotated[
        bool,
        typer.Option(
            "--l402",
            help="Fund via L402 sim (SEISO_PAY_L402_SIM=1 or faucet)",
        ),
    ] = False,
    json_out: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    """Create / status / fund marketplace sessions."""
    _require_allow()
    from seiso.pay.ark import funding_instructions
    from seiso.pay.flags import faucet_enabled
    from seiso.pay.l402 import complete_fund, l402_sim_enabled, mint_fund_challenge
    from seiso.pay.store import (
        activate_session,
        create_session,
        load_session,
        public_session_view,
        resolve_session_by_token,
    )

    act = action.strip().lower()
    if act == "create":
        created = create_session(scopes=[s.strip() for s in scopes.split(",")])
        tok = created["token"]
        sid = created["session_id"]
        if sats > 0 and l402:
            if not l402_sim_enabled():
                console.print(
                    "[red]SEISO_PAY_L402_SIM=1 (or SEISO_PAY_FAUCET=1) required "
                    "for L402 sim funding[/]"
                )
                raise typer.Exit(1)
            challenge = mint_fund_challenge(session_id=sid, amount_sats=sats)
            complete_fund(
                macaroon=str(challenge["macaroon"]),
                preimage_hex=str(challenge["sim_preimage"]),
            )
        elif sats > 0 and (faucet or faucet_enabled()):
            if not faucet_enabled():
                console.print("[red]SEISO_PAY_FAUCET=1 required for faucet funding[/]")
                raise typer.Exit(1)
            activate_session(sid, amount_sats=sats, funding_mode="faucet")
        record = load_session(sid)
        out = {
            "token": tok,
            "session": public_session_view(record),
            "funding": funding_instructions(sid, sats),
        }
        _print_json(out)
        return

    if act == "status":
        if token:
            record = resolve_session_by_token(token)
        elif session:
            record = load_session(session)
        else:
            env_tok = (os.environ.get("SEISO_PAY_TOKEN") or "").strip()
            if not env_tok:
                console.print("[red]Need --token, --session, or SEISO_PAY_TOKEN[/]")
                raise typer.Exit(1)
            record = resolve_session_by_token(env_tok)
        _print_json(public_session_view(record))
        return

    if act == "fund":
        if not session:
            console.print("[red]--session required[/]")
            raise typer.Exit(1)
        if sats <= 0:
            console.print("[red]--sats required[/]")
            raise typer.Exit(1)
        if l402:
            if not l402_sim_enabled():
                console.print(
                    "[red]SEISO_PAY_L402_SIM=1 (or SEISO_PAY_FAUCET=1) required "
                    "for L402 sim. Live Lightning is not wired.[/]"
                )
                _print_json(funding_instructions(session, sats))
                raise typer.Exit(2)
            try:
                challenge = mint_fund_challenge(session_id=session, amount_sats=sats)
                result = complete_fund(
                    macaroon=str(challenge["macaroon"]),
                    preimage_hex=str(challenge["sim_preimage"]),
                )
            except Exception as exc:
                console.print(f"[red]{exc}[/]")
                raise typer.Exit(1) from exc
            _print_json(result)
            return
        if not (faucet or faucet_enabled()):
            console.print(
                "[yellow]Ark funding: use funding.ark_address from session create. "
                "For L402 sim: --l402 with SEISO_PAY_L402_SIM=1. "
                "For faucet: --faucet with SEISO_PAY_FAUCET=1.[/]"
            )
            _print_json(funding_instructions(session, sats))
            raise typer.Exit(2)
        if not faucet_enabled():
            console.print("[red]SEISO_PAY_FAUCET=1 required[/]")
            raise typer.Exit(1)
        record = activate_session(session, amount_sats=sats, funding_mode="faucet")
        _print_json(public_session_view(record))
        return

    console.print(f"[red]Unknown action: {action}[/]")
    raise typer.Exit(1)


@pay_app.command("job")
def pay_job(
    action: Annotated[str, typer.Argument(help="start|status|list|cancel|receipt")],
    type_: Annotated[
        str | None, typer.Option("--type", help="Job type for start")
    ] = None,
    preset: Annotated[str | None, typer.Option()] = None,
    config: Annotated[str | None, typer.Option()] = None,
    session: Annotated[str | None, typer.Option("--session")] = None,
    job_id: Annotated[str | None, typer.Option("--id", "--job-id")] = None,
    dry_run: Annotated[bool, typer.Option(help="Escrow+settle without trainer")] = False,
) -> None:
    """Start or inspect marketplace jobs."""
    _require_allow()
    from seiso.pay.jobs import cancel_job, job_receipt, start_job
    from seiso.pay.store import list_jobs, load_job, resolve_session_by_token

    act = action.strip().lower()
    sid = session
    if not sid:
        tok = (os.environ.get("SEISO_PAY_TOKEN") or "").strip()
        if tok:
            sid = str(resolve_session_by_token(tok)["session_id"])

    if act == "start":
        if not type_:
            console.print("[red]--type required[/]")
            raise typer.Exit(1)
        if not sid:
            console.print("[red]--session or SEISO_PAY_TOKEN required[/]")
            raise typer.Exit(1)
        try:
            job = start_job(
                session_id=sid,
                job_type=type_,
                preset=preset,
                config=config,
                dry_run=dry_run,
            )
        except Exception as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        _print_json({"job": job, "receipt": job_receipt(job)})
        return

    if act == "list":
        if not sid:
            console.print("[red]--session or SEISO_PAY_TOKEN required[/]")
            raise typer.Exit(1)
        _print_json({"jobs": list_jobs(session_id=sid)})
        return

    if act in {"status", "receipt", "cancel"}:
        if not job_id:
            console.print("[red]--id required[/]")
            raise typer.Exit(1)
        if not sid:
            console.print(
                "[red]--session or SEISO_PAY_TOKEN required "
                "(must own the job, same as HTTP cancel/status)[/]"
            )
            raise typer.Exit(1)
        job = load_job(job_id)
        if str(job.get("session_id") or "") != str(sid):
            console.print("[red]job not found for this session[/]")
            raise typer.Exit(1)
        if act == "cancel":
            _print_json({"job": cancel_job(job_id)})
            return
        if act == "receipt":
            _print_json(job_receipt(job))
        else:
            _print_json({"job": job, "receipt": job_receipt(job)})
        return

    console.print(f"[red]Unknown action: {action}[/]")
    raise typer.Exit(1)


@pay_app.command("serve")
def pay_serve(
    host: Annotated[str, typer.Option(help="Bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port")] = 8787,
) -> None:
    """Run the marketplace sidecar (operator). Put TLS proxy in front for public."""
    _require_allow()
    from seiso.pay.app import run_server
    from seiso.pay.flags import protocol_treasury_ark

    if not protocol_treasury_ark():
        console.print(
            "[yellow]Warning: SEISO_PROTOCOL_TREASURY_ARK unset — "
            "paid settles fail-closed unless SEISO_PAY_FAUCET=1[/]"
        )
    console.print(f"Starting pay sidecar on http://{host}:{port}")
    run_server(host=host, port=port)
