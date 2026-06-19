"""Content-Security-Policy and security header helpers for Forge."""

from __future__ import annotations

# Inline scripts allowed by local tooling (Vite React refresh preamble, etc.).
# Browsers may suggest additional hashes in CSP violation reports.
SCRIPT_HASHES: tuple[str, ...] = (
    # Vite + @vitejs/plugin-react dev preamble (npm run dev).
    "sha256-Z2/iFzh9VMlVkEOar1f/oSHWwQk3ve1qk/C2WdsC4Xk=",
    # Reported by Firefox/Chrome when Forge CSP blocks a bundled inline helper.
    "sha256-ZswfTY7H35rbv8WC7NXBoiC7WNu86vSzCDChNWwZZDM=",
)

# Common loopback services when Forge is bound to 127.0.0.1 (UI, Vite, Ollama, vLLM).
_LOCAL_CONNECT_SRC: tuple[str, ...] = (
    "http://127.0.0.1:8765",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
    "http://127.0.0.1:11434",
    "http://127.0.0.1:8000",
    "http://localhost:8765",
    "http://localhost:5173",
    "http://localhost:4173",
    "http://localhost:11434",
    "http://localhost:8000",
    "ws://127.0.0.1:*",
    "ws://localhost:*",
    "wss://127.0.0.1:*",
    "wss://localhost:*",
)


def is_document_path(path: str) -> bool:
    """True for SPA shell routes that execute scripts in the browser."""
    if path.startswith(("/api/", "/v1/", "/assets/")):
        return False
    return path not in {"/health", "/api/health"}


def build_csp_policy(*, nonce: str | None = None, local_only: bool = True, debug: bool = False) -> str:
    """Return a CSP header value tuned for local-first vs remote exposure."""
    script_elem = ["'self'", *SCRIPT_HASHES]
    script_src = ["'self'", *SCRIPT_HASHES]
    if nonce:
        script_elem.append(f"'nonce-{nonce}'")
        script_src.append(f"'nonce-{nonce}'")
    if debug:
        # Swagger (/api/docs) and Vite HMR inject small inline helpers.
        script_elem.append("'unsafe-inline'")
        script_src.append("'unsafe-inline'")

    connect_src = ["'self'"]
    if local_only:
        connect_src.extend(_LOCAL_CONNECT_SRC)
    if debug:
        connect_src.extend(("ws:", "wss:"))

    return (
        "default-src 'self'; "
        "base-uri 'self'; "
        "object-src 'none'; "
        "form-action 'self'; "
        f"connect-src {' '.join(connect_src)}; "
        "style-src 'self' 'unsafe-inline'; "
        f"script-src {' '.join(script_src)}; "
        f"script-src-elem {' '.join(script_elem)}; "
        "script-src-attr 'none'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        "media-src 'self' blob:; "
        "worker-src 'self' blob:; "
        "frame-ancestors 'none'"
    )


def apply_response_security_headers(
    *,
    path: str,
    response_headers: dict[str, str],
    local_only: bool,
    debug: bool,
    existing_csp: str | None = None,
) -> None:
    """Apply baseline security headers with local-first vs remote-aware CSP."""
    response_headers.setdefault("X-Content-Type-Options", "nosniff")
    response_headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response_headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")

    if path.startswith("/assets/"):
        response_headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        # Static bundles are public; CORP same-origin is unnecessarily strict for local tooling.
        response_headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        return

    response_headers.setdefault("X-Frame-Options", "DENY")
    response_headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response_headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")

    if path.startswith(("/api/", "/v1/")):
        # JSON/SSE API responses do not need script CSP.
        return

    if existing_csp:
        response_headers["Content-Security-Policy"] = existing_csp
        return

    if is_document_path(path):
        response_headers["Content-Security-Policy"] = build_csp_policy(
            local_only=local_only,
            debug=debug,
        )
