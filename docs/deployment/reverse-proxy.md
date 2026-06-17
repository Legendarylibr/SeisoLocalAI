# HTTPS reverse proxy

Terminate TLS in front of Seiso Forge so you can access it over **HTTPS** without changing Forge’s local-first defaults. Forge keeps listening on `127.0.0.1:8765`; the reverse proxy handles certificates and public exposure.

## When to use this

| Scenario | Reverse proxy? |
|----------|----------------|
| Access Forge UI over HTTPS (remote or LAN) | Yes — recommended |
| Secure session cookies over HTTPS | Yes — set `SEISO_SECURE_COOKIES=true` |
| HTTP-only **public** LLM API as a provider | Yes — see [Provider proxy](#https-wrapper-for-http-llm-apis) |
| Local Ollama / vLLM | No — use `http://127.0.0.1:11434` or `:8000` directly |
| LAN/private IP LLM server | No — SSRF rules still block private ranges |

## 1. Configure Forge

Copy the example env and edit your domain:

```bash
cp deploy/env.https.example .env
```

```bash
SEISO_HOST=127.0.0.1
SEISO_PORT=8765
SEISO_ALLOW_REMOTE=false      # keep Forge on localhost
SEISO_TRUST_PROXY=true        # honor X-Forwarded-* from the proxy
SEISO_SECURE_COOKIES=true     # Secure flag on session/CSRF cookies
SEISO_CORS_ORIGINS=https://forge.example.com
```

| Variable | Purpose |
|----------|---------|
| `SEISO_TRUST_PROXY` | Rate limits and login throttling use the real client IP from `X-Forwarded-For` |
| `SEISO_SECURE_COOKIES` | Marks cookies `Secure` when TLS is terminated by the proxy (without opening LAN bind) |
| `SEISO_CORS_ORIGINS` | Must include your **https://** public URL |

Start Forge (from repo root, after building the UI):

```bash
cd forge-ui && npm install && npm run build && cd ..
seiso forge
```

## 2. Option A — Caddy (recommended)

Edit `deploy/caddy/Caddyfile` — replace `forge.example.com` with your domain.

**Native install:**

```bash
sudo cp deploy/caddy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl enable --now caddy
sudo systemctl reload caddy
```

Caddy obtains Let’s Encrypt certificates automatically.

**Docker** (Forge on host, Caddy in container):

```bash
# Edit deploy/caddy/Caddyfile.docker domain first
cd deploy
docker compose -f docker-compose.caddy.yml up -d
```

## 3. Option B — nginx

```bash
sudo cp deploy/nginx/seiso-forge.conf /etc/nginx/sites-available/seiso-forge
# Edit server_name and ssl_certificate paths
sudo ln -s /etc/nginx/sites-available/seiso-forge /etc/nginx/sites-enabled/
sudo certbot --nginx -d forge.example.com
sudo nginx -t && sudo systemctl reload nginx
```

The nginx config disables buffering for **SSE** streams (training logs, chat, downloads).

## 4. systemd (optional)

Run Forge as a service:

```bash
sudo useradd -r -s /usr/sbin/nologin seiso
sudo mkdir -p /opt/seiso
# Copy repo + venv to /opt/seiso, set .env
sudo cp deploy/systemd/seiso-forge.service /etc/systemd/system/
sudo systemctl enable --now seiso-forge
```

Pair with Caddy or nginx on the same host.

## Architecture

```text
Browser ──HTTPS──► Caddy/nginx (:443)
                        │
                        └──HTTP──► Seiso Forge (127.0.0.1:8765)
```

Forge never needs direct TLS support. The proxy forwards:

- `Host`
- `X-Real-IP` / `X-Forwarded-For` (for rate limits when `SEISO_TRUST_PROXY=true`)
- `X-Forwarded-Proto` (for uvicorn proxy headers)

## HTTPS wrapper for HTTP LLM APIs

Seiso requires **HTTPS** for remote provider URLs (except local Ollama/vLLM on loopback). If you have an HTTP-only public API, put it behind HTTPS:

```bash
# deploy/caddy/Caddyfile.provider-proxy
llm-proxy.example.com {
    reverse_proxy http://upstream-api-host:8080
}
```

In Forge **Settings → Providers**, set:

```text
base_url: https://llm-proxy.example.com/v1
```

**Limits:** This only fixes the HTTP→HTTPS scheme check. Private IPs (`192.168.x.x`, `10.x.x.x`), localhost, and metadata hosts are still blocked by SSRF protection.

## Troubleshooting

### Login works on HTTP but not HTTPS

- Set `SEISO_SECURE_COOKIES=true`
- Add your HTTPS origin to `SEISO_CORS_ORIGINS`
- Clear browser cookies for the old HTTP origin

### CSRF validation failed

`SEISO_CORS_ORIGINS` must exactly match the browser origin (scheme + host + port).

### Rate limit hits everyone at once

Set `SEISO_TRUST_PROXY=true` and ensure the proxy sends `X-Forwarded-For`.

### SSE streams stall (training / chat)

- **nginx:** `proxy_buffering off` (included in `deploy/nginx/seiso-forge.conf`)
- **Caddy:** `flush_interval -1` (included in Caddyfile)

### Provider URL rejected

| Error | Fix |
|-------|-----|
| `base_url must use HTTPS` | Use HTTPS URL or local Ollama/vLLM on loopback |
| `resolves to a blocked network range` | Reverse proxy cannot expose private IPs to Forge |

See also [troubleshooting.md](../troubleshooting.md).
