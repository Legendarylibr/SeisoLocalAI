# Deployment configs

Ready-to-use configs for HTTPS access to Seiso Forge.

| File | Purpose |
|------|---------|
| [env.https.example](env.https.example) | `.env` values for reverse-proxy deployment |
| [caddy/Caddyfile](caddy/Caddyfile) | Caddy on host (proxy → `127.0.0.1:8765`) |
| [caddy/Caddyfile.docker](caddy/Caddyfile.docker) | Caddy in Docker (proxy → host) |
| [caddy/Caddyfile.provider-proxy](caddy/Caddyfile.provider-proxy) | HTTPS wrapper for HTTP-only LLM APIs |
| [nginx/seiso-forge.conf](nginx/seiso-forge.conf) | nginx TLS termination + SSE |
| [docker-compose.caddy.yml](docker-compose.caddy.yml) | Run Caddy in Docker |
| [systemd/seiso-forge.service](systemd/seiso-forge.service) | systemd unit for Forge |

Full guide: [docs/deployment/reverse-proxy.md](../docs/deployment/reverse-proxy.md)

## Quick start (Caddy on host)

```bash
cp deploy/env.https.example .env
# Edit SEISO_CORS_ORIGINS and deploy/caddy/Caddyfile domain

seiso forge

# Install Caddy, then:
sudo cp deploy/caddy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Browse to `https://forge.example.com`.
