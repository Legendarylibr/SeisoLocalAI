# TUI agent harnesses

Configure optional coding-agent CLIs **inside the existing `seiso tui`** (Settings + Integrations + Chat slashes). There is no second TUI and Chat Enter still sends a normal offline message.

## Harnesses

| Id | CLI | Role in v1 |
|---|---|---|
| `hermes` | `hermes` | Default worker when installed |
| `pi` | `pi` | Headless `--mode json -p` |
| `omp` | `omp` | Same Pi-family adapter |
| `cline` | `cline` | Detect + optional oneshot |
| `openclaw` | `openclaw` / `clawdbot` | Detect + optional oneshot |

Seiso never rewrites `~/.hermes`, `~/.pi`, `~/.openclaw`, or other user configs. Isolated snippets land under `$SEISO_DATA_DIR/agent/harnesses/<id>/`.

## Settings

- Cycle harness, model source (auto / Ollama / Smart Router / Forge), route class
- **Seiso subagents** master switch (default **off** — worker only, no extra work)
- Turning **on** enables the cheap pair: completion + correctness (syntax compile of files the worker touched). No extra model load unless you turn `allow_llm` on for a role.
- Planner / synthesizer stay off until you enable them. Planner output is prepended to the worker goal.

## Chat slashes

- `/harness pi|omp|hermes|cline|openclaw`
- `/subagents on|off`
- `/swarm single|pair|plan_act_verify`
- `/agent <goal>` — headless swarm into the current Chat transcript

## Resource limits

Subagents default off. When on, roles run **sequentially**. Completion/correctness improve quality by catching failed exits and broken Python **without** loading another model. An optional LLM judge is skipped with `blocked:oom_guard` if VRAM/RAM headroom is too small.
