# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities **privately** through GitHub's private
vulnerability reporting — do **not** open a public issue or pull request for
anything security-sensitive.

**[Report a vulnerability →](https://github.com/Legendarylibr/SeisoLocalAI/security/advisories/new)**

(GitHub → repository **Security** tab → **Advisories** → **Report a vulnerability**.)

Reports filed this way are visible only to the maintainers until a fix is
released and a coordinated advisory is published.

### What to include

- A description of the issue and its impact (what an attacker gains).
- Affected component (Forge API, CLI, training/export pipeline, bundled
  research code, deploy configs, ...) and version/commit.
- Reproduction steps or a proof of concept, ideally against a default
  localhost install.
- Any relevant configuration (e.g. `SEISO_ALLOW_REMOTE`, `SEISO_ALLOW_TOOLS`,
  `SEISO_ALLOW_CODE_EXEC` values), since much of Seiso's attack surface is
  opt-in.

### What to expect

- Acknowledgement of your report within **7 days**.
- A triage decision (accepted / needs info / declined) within **14 days**.
- Coordinated disclosure: we will work with you on a fix and publish a GitHub
  security advisory (with credit, if you want it) once a patched release is
  available. Please keep the report private until then.

## Supported versions

Seiso is in alpha. Only the latest release and the current `main` branch
receive security fixes.

| Version | Supported |
|---------|-----------|
| `main` / latest release | Yes |
| Older releases | No |

## Scope notes

Seiso is **secure by default for single-user localhost use** (see
[README § Security](README.md#security)). Reports are especially valuable for:

- Sandbox escapes: path traversal out of `SEISO_DATA_DIR`, cross-user access
  between per-user directories.
- Auth/session flaws, CSRF, or rate-limit bypasses in the Forge API.
- SSRF or DNS-rebinding bypasses in outbound provider calls.
- Leakage of encrypted-at-rest secrets (HF tokens, DB columns).
- Code execution outside the opt-in sandboxed tools.

Issues that require the user to explicitly disable documented safeguards
(e.g. setting `SEISO_ALLOW_REMOTE=true` without a reverse proxy, or acking
`SEISO_REMOTE_DANGEROUS_ACK`) are generally treated as hardening guidance
rather than vulnerabilities, but we still want to hear about them.

Vulnerabilities in third-party dependencies should go to the upstream project;
open a report here only if Seiso's usage of the dependency is what makes it
exploitable.
