# Seiso roadmap

**North star:** Seiso becomes a **local agentic operating system** — a harness on your machine that plans and runs work, routes each step to the right model (local first, external router when useful), and can optionally buy or sell **distributed inference and training** for **Bitcoin / Lightning / other crypto** payments.

This is the destination, not a promise of dates. Self-hosted Forge and the CLI stay **free**. There is **no Seiso token**. Marketplace and mesh remain **opt-in**.

| Today | Destination |
|-------|-------------|
| Local-first workspace: chat, train, compress, export | Same workspace as the **control plane** of a personal AI OS |
| Agent skill + tool loop + receipts | First-class **harness** (plan → route → act → verify → receipt) |
| Local backends + optional [SeisoModelRouter](https://github.com/Legendarylibr/SeisoModelRouter) | **Model-aware routing**: local models by default, external router for hard choices |
| Pay / mesh scaffolding | **Marketplace** for remote inference and training, settled in BTC / sats / crypto |

Related docs: [getting started](docs/getting-started.md) · [agent skill](.agents/skills/seiso-orchestrate/SKILL.md) · [marketplace](docs/pay/marketplace.md) · [mesh](docs/training/mesh.md) · [Smart Router](https://github.com/Legendarylibr/SeisoModelRouter)

---

## Why an OS, not just a chat app

A local model UI is not enough. The useful unit of work is an **agent session** that can:

1. Inspect hardware, models, datasets, and running jobs on *this* machine.
2. Choose **which model** should handle a step (small local GGUF vs larger local specialist vs routed remote).
3. Call tools (search, sandboxed code, train/export/compress) under the existing policy flags.
4. Spill over to **trusted mesh peers** or a **paid marketplace** only when local capacity is not enough.
5. Leave an auditable receipt (job ids, model used, sats split) — never secrets.

That is the operating system: **scheduler + policy + router + wallets**, with your GPU as the default computer.

```
 You / IDE / Buzz room
          │
          ▼
 ┌──────────────────────────────────────────┐
 │  Local agentic OS (this repo)            │
 │  Forge · TUI · CLI · Compat /v1          │
 │                                          │
 │  Harness  ──►  model-aware router  ──►   │
 │  plan,        local backends +           │
 │  tools,       optional external router   │
 │  receipts                                │
 └────────────┬─────────────────┬───────────┘
              │                 │
     local GPU / RAM            │  only if you opt in
              │                 │
              ▼                 ▼
     train · infer ·      ┌─────────────┐   ┌─────────────────────┐
     compress · export    │ Buzz mesh   │   │ Marketplace         │
                          │ trusted     │   │ distributed infer + │
                          │ peers, $0   │   │ train, BTC / LN /   │
                          └─────────────┘   │ Ark / L402 / crypto │
                                            └─────────────────────┘
```

**Compute decision (unchanged):** local free → mesh peers (no fee) → paid marketplace → ask a human. Never charge localhost.

---

## Pillar 1 — Local agentic operating system

Seiso already owns the machine-local control plane: jobs, SSE logs, memory guards, path sandbox, Nostr identity, Compat API. The OS layer is that control plane grown into something agents can live in.

| Layer | Role |
|-------|------|
| Surfaces | Forge UI, `seiso tui`, `seiso` CLI, `/v1` for Cursor / other clients |
| Kernel-ish | Job orchestrators, hardware/VRAM guards, path sandbox, auth |
| Policy | Opt-in tools, code-exec, remote bind, pay, mesh — all default **off** |
| Identity | Local Nostr `npub` / `nsec`; provenance attestations |
| Persistence | `SEISO_DATA_DIR` (`~/.seiso`) — models, checkpoints, knowledge, receipts |

**Now**

- Single-user localhost Forge + CLI + TUI
- Isolated GGUF chat sidecar on Linux NVIDIA (Ollama first, llama-swap fallback)
- Job runners for train / slime / NeMo RL / export / compress / distill-RL / knowledge
- Compat `/v1` so external agents can talk to local models
- Compute kernel: `decide_compute` (`seiso/agent/kernel.py`) — local → mesh → pay → `ask_human`

**Next**

- Long-lived agent sessions as first-class objects (not one-shot chat turns)
- A permission graph: which agent may train, spend sats, or launch mesh
- Scheduled / background agents (watch a dataset, resume a job, attest a digest)
- Stronger process isolation for tools (beyond the AST sandbox)

**Later**

- Multi-agent rooms on one machine (researcher, trainer, reviewer) sharing the same policy kernel
- Device and model inventory as OS resources agents can query and lock
- Optional multi-user tenants that still never phone home

---

## Pillar 2 — Harness

The harness is how an agent *drives* the OS: observe, plan, call Seiso, verify, write a receipt.

**Now**

- [`.agents/skills/seiso-orchestrate/`](.agents/skills/seiso-orchestrate/SKILL.md) — generic coding-agent / Buzz skill
- `SEISO_AGENT=1` / `SEISO_TRAINING_SURFACE=agent` vs Forge **frontend** surface
- `seiso agent status` receipts; Buzz is optional (mesh signing still needs a Buzz `nsec`)
- Opt-in chat tool loop (`forge/tools/agent_loop.py`): search, sandboxed `execute_code`, artifacts
- Agents prefer CLI over Forge HTTP; `--launch` on mesh is human-gated
- In-tree harness: `run_harness` (`seiso/agent/harness.py`) — plan → decide → `select_route` → act → verify → receipt
- CLI: `seiso agent decide`, `seiso agent plan --dry-run`

**Next**

- Structured tool schema for train / export / compress / pay / mesh (JSON, not prompt-only)
- Evaluation harness: replay a plan against smoke configs and score artifacts
- Memory for the agent (job history, preferred models, last good presets) stored under `SEISO_DATA_DIR`

**Later**

- Multi-step programs (“distill this teacher, QLoRA the student, export GGUF, attest, list on the market”)
- Cross-machine harness: one planner, many Seiso workers (mesh or marketplace)
- Formal receipts (Nostr + optional Lean membership proofs) as the audit log of the OS

---

## Pillar 3 — Model-aware routing (local + external)

Routing is **model-aware**: pick a backend and a checkpoint from the task, context length, VRAM, latency, and (only if opted in) price. Local weights win when they fit. An **external router** is a specialist you run beside Forge — Seiso does not have to host inference for you.

**Now**

- Local backends: llama.cpp / llama-swap / Ollama, MLX, PyTorch ([docs/inference/backends.md](docs/inference/backends.md))
- Context-window and memory-aware load clamps
- Chat **Smart Router (auto-route)** when `SEISO_MODEL_ROUTER_ENABLED=true` points at a localhost router such as [SeisoModelRouter](https://github.com/Legendarylibr/SeisoModelRouter) (`/v1/chat/completions`)
- Provider integrations (OpenAI, Anthropic, vLLM) with SSRF hardening — optional, not the default path
- Local picker: `select_route` (`seiso/routing/select.py`) — VRAM / context / role, step-down, localhost-only external router
- CLI: `seiso route --task chat`

**Next**

- Wire `select_route` into Forge Chat / Compat `/v1` so the UI uses the same picker
- Live inventory from `SEISO_DATA_DIR` / Hub cache (CLI still takes `--inventory-json`)

**Later**

- Hybrid turns: cheap local model for tools and classification; external router only for the hard generation step
- Marketplace quotes as just another route cost (sats / token, latency, attested GPU)

External routers stay **replaceable**. SeisoModelRouter is the reference; any localhost OpenAI-compatible router is fine.

---

## Pillar 4 — Marketplace for distributed inference and training

When this machine cannot (or should not) run the job, the OS can buy **inference** or **training** from operators, and operators can sell spare GPU time. Settlement is **Bitcoin / Lightning / crypto** — not a platform coin.

**Now (scaffolding — do not use for real funds)**

- Opt-in sidecar `seiso pay` ([docs/pay/marketplace.md](docs/pay/marketplace.md))
- Job types in the quote API: inference, finetune, RL / slime
- Designed rails: **Ark** (operator + protocol treasury split) and **L402** (Lightning HTTP 402)
- Faucet / simulated ledger only; live Ark and live Lightning are **not wired**
- Default **5%** protocol fee on top of operator price; localhost is never billed
- Experimental **Buzz mesh** for trusted peers with **no** protocol fee ([docs/training/mesh.md](docs/training/mesh.md))
- Catalog type `Listing` + `quote_listing` (`seiso/pay/catalog.py`) — inference / finetune / slime / distill_rl / nemo_rl; **no Seiso token**; loopback listings are 0 sats

**Next**

- Live Lightning L402 (real invoices + preimage) for prepaid sessions
- Live Ark (or equivalent BTC rail) pay-in and fee split, fail-closed without treasury
- Operator catalog: models offered, GPU class, queue, advertised $/sat rates
- Buyer flow from the harness: quote → fund → run `/v1` or `seiso pay job` → receipt
- Same job types as local: chat completions, LoRA/QLoRA, slime, distill-RL, export artifacts

**Later**

- Open marketplace: many operators, discovery, reputation, slashing / refunds on attested failure
- Distributed **inference** (token streaming from remote GPUs) and **training** (data stays hashed; artifacts come back)
- Additional crypto rails if operators need them (on-chain BTC, other Lightning, later non-BTC crypto) — still **no Seiso token**
- Mesh peers may *also* list on the market; pay is for strangers, mesh is for people you already trust
- Nostr provenance on delivered checkpoints so buyers can verify what they paid for

Public operators expose the **pay sidecar + TLS**, not Forge. `SEISO_ALLOW_REMOTE` stays off unless the operator really means it.

---

## Suggested sequence

Order is about dependency, not calendar quarters.

| Phase | Focus | Exit when |
|-------|--------|-----------|
| **0 — Harden the local OS** | Surfaces, jobs, memory, security, Compat `/v1` | Daily chat + train + export is boringly reliable |
| **1 — Harness** | In-tree plan/act/receipt loop; structured tools | An agent can run a smoke train → export → attest without a human clicking Forge |
| **2 — Model-aware routing** | Local table + external router as one API | Harness gets a reasoned `{backend, model}` for chat and jobs |
| **3 — Wire payments** | Live L402 and/or Ark; faucet stays for CI | A buyer can pay real sats for a remote smoke job end-to-end |
| **4 — Market** | Listings for inference + training; discovery; receipts | Two operators and one harness client complete a paid job |
| **5 — Distributed scale** | Mesh + market together; attested artifacts | Spare GPUs on a LAN or the public net are just more OS devices |

Phase 0 is largely this repository today. Phases 3–4 must stay **opt-in** and fail closed. Until settlement is live, treat [docs/pay/marketplace.md](docs/pay/marketplace.md) as design, not a product.

---

## Non-goals

- A Seiso coin, airdrop, or mandatory token to use local features
- Charging people to run Forge or the CLI on their own hardware
- Replacing [SeisoModelRouter](https://github.com/Legendarylibr/SeisoModelRouter) with a hosted inference business inside this repo
- Making Buzz or any single relay a hard dependency of local work
- Shipping live payments before refunds, escrow, and treasury split are fail-closed

---

## How to follow along

- Track implementation in GitHub issues and PRs; this file is updated when a pillar’s “now / next” actually moves.
- Local work: [docs/README.md](docs/README.md)
- Agent loops: [AGENTS.md](AGENTS.md) and the [orchestrate skill](.agents/skills/seiso-orchestrate/SKILL.md)
- Payments: [docs/pay/marketplace.md](docs/pay/marketplace.md) — **not functional for real funds today**
- Routing: [docs/cli.md § External Smart Router](docs/cli.md#external-smart-router) and Forge env `SEISO_MODEL_ROUTER_*`
