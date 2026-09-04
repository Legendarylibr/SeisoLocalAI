# Skills & Agents Reference — Seiso

Consolidated inventory of skills and agent definitions available when working on the Seiso project (local-first AI platform for training, inference, quantization, compression, and publishing).

---

## Hermes Skills

### Software Development

| Skill | Purpose |
|---|---|
| Hermes Agent Skill Authoring | Author in-repo SKILL.md: frontmatter, validator, structure, writing quality. |
| Node Inspect Debugger | Debug Node.js via `--inspect` + Chrome DevTools Protocol CLI. |
| Plan | Write actionable markdown implementation plans to `.hermes/plans/`. |
| Python Debugpy | Debug Python via `breakpoint()`+pdb or remote debugpy (DAP protocol). |
| Requesting Code Review | Pre-commit verification: security scan, quality gates, independent reviewer. |
| Simplify Code | Parallel 3-agent (reuse/quality/efficiency) cleanup of recent changes. |
| Spike | Throwaway experiments to validate feasibility before committing. |
| Systematic Debugging | 4-phase root-cause debugging: investigate, pattern analysis, hypothesis, fix. |
| Test-Driven Development | Enforce RED-GREEN-REFACTOR cycle: tests before code, no exceptions. |

### GitHub

| Skill | Purpose |
|---|---|
| Codebase Inspection | Analyze repos: LOC, language breakdown, file counts, code-vs-comment ratios (pygount). |
| GitHub Auth | Auth setup: HTTPS tokens, SSH keys, `gh` CLI login. Prerequisite for other GitHub skills. |
| GitHub Code Review | Code review of diffs/commits — inline comments, formal submissions. |
| GitHub Issues | Create, search, triage, label, assign, manage issues via `gh` or REST API. |
| GitHub PR Workflow | Full PR lifecycle: branch, commit, open, monitor CI, auto-fix failures, merge. |
| GitHub Repo Management | Clone, create, fork, configure repos; remotes, releases, secrets, branch protection. |

### MLOps

#### Evaluation

| Skill | Purpose |
|---|---|
| LM Evaluation Harness | Benchmark LLMs on 60+ academic benchmarks (MMLU, GSM8K, HumanEval) via lm-eval. |
| Weights & Biases | ML experiment tracking, sweeps, model registry, artifact lineage, dashboards. |

#### Hub

| Skill | Purpose |
|---|---|
| Hugging Face Hub | Modern `hf` CLI: models, datasets, Spaces, repos, SQL queries. |

#### Inference

| Skill | Purpose |
|---|---|
| llama.cpp | Local GGUF inference (CPU/GPU) + Hub discovery for quantized models. |
| vLLM | High-throughput LLM serving, OpenAI API, quantization, tensor parallelism, monitoring. |

#### Models

| Skill | Purpose |
|---|---|
| AudioCraft | Meta AudioCraft: text-to-music (MusicGen) and text-to-sound (AudioGen). |
| Segment Anything | Meta SAM: zero-shot image segmentation via points, boxes, or mask prompts. |

### Data Science

| Skill | Purpose |
|---|---|
| Jupyter Live Kernel | Iterative Python via live Jupyter kernel (hamelnb) — stateful REPL for data science exploration. |

### Dogfood / QA

| Skill | Purpose |
|---|---|
| Dogfood | Systematic exploratory QA testing of web apps: find bugs, collect evidence, structured reports. |

### Research

| Skill | Purpose |
|---|---|
| arXiv | Search arXiv papers + Semantic Scholar for citations/related work. |
| Research Paper Writing | End-to-end ML paper pipeline: literature review, experiments, drafting, submission. |

### Productivity

| Skill | Purpose |
|---|---|
| Nano PDF | Edit PDF text/typos/titles via natural-language CLI instructions. |
| OCR & Documents | Extract text from PDFs/scans (pymupdf lightweight, marker-pdf for OCR/equations). |

### Autonomous AI Agents

| Skill | Tool | Purpose |
|---|---|---|
| Claude Code | `claude` CLI | Delegate coding — one-shot (print mode) or interactive PTY/tmux sessions. |
| Codex | `codex` CLI | Delegate coding to OpenAI Codex CLI. Git repo required. Exec, background tasks, PR reviews. |
| Hermes Agent | `hermes` | Configure/extend/contribute to Hermes Agent: CLI, profiles, credentials, MCP, gateways, skills, cron. |
| OpenCode | `opencode` CLI | Delegate coding — provider-agnostic. One-shot or interactive TUI. |

### Computer Use

| Skill | Tool | Purpose |
|---|---|---|
| Computer Use | `cua-driver` | Drive desktop in background: click, type, scroll. Cross-platform, no focus steal. |

### Creative (Seiso-relevant subset)

| Skill | Purpose |
|---|---|
| Sketch | Disposable HTML mockups — 2-3 design variants for side-by-side comparison (UI prototyping for Forge). |
| Architecture Diagram | Dark-themed architecture/infra diagrams as standalone HTML + inline SVG. No external deps. |

---

## Codex System Skills

Six built-in skills, each paired with an agent definition.

| Skill | Agent | Purpose |
|---|---|---|
| Image Gen | `imagegen/agents/openai.yaml` | Generate/edit images via built-in tool + CLI fallback (gpt-image-1.5). Chroma-key removal, batch generation, prompt augmentation. |
| OpenAI Docs | `openai-docs/agents/openai.yaml` | OpenAI API docs, model selection/migration, prompt-upgrade guidance. MCP server `openaiDeveloperDocs` (streamable HTTP). |
| Plugin Creator | `plugin-creator/agents/openai.yaml` | Scaffold plugin directories with `plugin.json`, marketplace entries, cachebuster/reinstall flow, validation. |
| Review Agent | `review-agent/agents/openai.yaml` | Read-only code review of diffs/commits — P0-P3 findings. Never modifies files. |
| Skill Creator | `skill-creator/agents/openai.yaml` | Build modular SKILL.md folders with resources, progressive disclosure, naming conventions. 6-step process. |
| Skill Installer | `skill-installer/agents/openai.yaml` | Install skills from curated list, experimental list, or any GitHub repo via download or git sparse-checkout. |

---

## Cursor Skills

| Skill | Purpose |
|---|---|
| AI Redteam Ultra | Ethical adversarial red-team security audit of LLM/agent systems. 11-phase methodology covering prompt injection, RAG poisoning, trust boundaries, structural hardening. Reference files: analysis-phases, engagement-modes, exploit-chains, hardening-principles, recon-checklist, severity-rubric, target-profiles, threat-domains, trust-boundary-diagrams. |

---

## Repo-bundled Skills

### Seiso Orchestrate (Buzz integration)

`seiso-orchestrate` is a skill for orchestrating Seiso (Forge + CLI) from a Buzz agent room using `buzz-cli` for channel updates and Seiso for local train/compress/export/provenance jobs.

- **Location:** this section of `AGENTS.md`
- **Preconditions:** `BUZZ_PRIVATE_KEY` (agent nsec), `BUZZ_RELAY_URL`, Seiso install paths (`SEISO_INSTALL_DIR`, `SEISO_DATA_DIR`)
- **Workflow:** Join/create Buzz channel → set topic → run Seiso CLI jobs → post receipts (commands, job IDs, manifest paths, Nostr `event_id`) back to channel
- **Supported jobs:** `seiso train`, `seiso compress`, `seiso distill-rl`, `seiso rl-quant`, `seiso export`, `seiso provenance attest|verify`
- **Safety:** Never post `nsec`, HF tokens, backup passphrases, or cookie headers to Buzz. Never automate Forge keygen. Default Forge bind is localhost.
- **Decision guide:** Smoke configs for quick iteration, Forge UI when human is watching, CLI for agent loops
- **Compute decision:** Call `decide_compute` (`seiso.agent.kernel`) or `seiso agent decide` instead of re-implementing the local → mesh → pay → ask_human order. Never point `SEISO_PAY_URL` at localhost — `decide_compute` refuses loopback pay.

---

## Seiso integration map

| Skill | Seiso component |
|---|---|
| llama.cpp | GGUF inference backend, model loading profiles (`~/.seiso/cache/llama_load_profiles.json`) |
| vLLM | High-throughput serving, slime RL rollouts |
| Hugging Face Hub | Model downloads, dataset access, publishing (`~/.seiso/hf_cache/`) |
| GitHub PR Workflow | CI/CD pipeline, PR lifecycle for contributions |
| LM Evaluation Harness | Benchmarking trained models |
| Weights & Biases | Experiment tracking for training runs |
| Segment Anything | Image segmentation integration |
| AudioCraft | Audio generation integration |
| Computer Use | Desktop automation for Forge UI testing |
| Sketch | UI prototyping for Forge |
| Architecture Diagram | Infra documentation |

### Documentation

Key reference files in the Seiso docs tree:

- `docs/README.md` — Full doc index + learning paths
- `docs/getting-started.md` — Quickstart guide
- `docs/install.md` — Installation reference
- `docs/forge.md` — Forge dev mode with hot reload
- `docs/ANALYSIS.md` — Architecture overview, feature map, code health
- `docs/CI_LOCAL.md` — CI quality gate
- `docs/provenance-nostr.md` — Nostr digest attestation system
- `docs/compression.md` — LLM compression pipeline
- `docs/cli.md` — CLI reference
- `docs/troubleshooting.md` — Troubleshooting guide
- `CONTRIBUTING.md` — Contribution guidelines
- `SECURITY.md` — Security policy

