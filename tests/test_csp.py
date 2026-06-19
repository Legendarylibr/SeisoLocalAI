from __future__ import annotations

from forge.security.csp import (
    SCRIPT_HASHES,
    apply_response_security_headers,
    build_csp_policy,
    is_document_path,
)


def test_build_csp_policy_includes_reported_script_hashes():
    policy = build_csp_policy()
    for digest in SCRIPT_HASHES:
        assert digest in policy
    assert "script-src-elem 'self'" in policy


def test_build_csp_policy_uses_nonce_for_spa_shell():
    policy = build_csp_policy(nonce="abc123")
    assert "'nonce-abc123'" in policy


def test_local_only_allows_loopback_connect():
    policy = build_csp_policy(local_only=True)
    assert "http://127.0.0.1:8765" in policy
    assert "ws://127.0.0.1:*" in policy
    assert "font-src 'self' data:" in policy


def test_remote_mode_keeps_connect_src_tight():
    policy = build_csp_policy(local_only=False)
    assert "connect-src 'self';" in policy
    assert "127.0.0.1:5173" not in policy


def test_build_csp_policy_allows_vite_hmr_in_debug():
    policy = build_csp_policy(debug=True)
    assert "'unsafe-inline'" in policy
    assert "connect-src 'self'" in policy
    assert "ws:" in policy


def test_is_document_path():
    assert is_document_path("/")
    assert is_document_path("/chat")
    assert not is_document_path("/api/inference/chat")
    assert not is_document_path("/assets/index.js")
    assert not is_document_path("/v1/chat/completions")


def test_apply_response_security_headers_skips_csp_on_api():
    headers: dict[str, str] = {}
    apply_response_security_headers(
        path="/api/inference/models",
        response_headers=headers,
        local_only=True,
        debug=False,
    )
    assert "content-security-policy" not in {k.lower() for k in headers}
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_apply_response_security_headers_sets_csp_on_spa():
    headers: dict[str, str] = {}
    apply_response_security_headers(
        path="/chat",
        response_headers=headers,
        local_only=True,
        debug=False,
    )
    assert "Content-Security-Policy" in headers
    assert "X-Frame-Options" in headers


def test_apply_response_security_headers_static_assets_use_cross_origin_corp():
    headers: dict[str, str] = {}
    apply_response_security_headers(
        path="/assets/index.js",
        response_headers=headers,
        local_only=True,
        debug=False,
    )
    assert headers["Cross-Origin-Resource-Policy"] == "cross-origin"
    assert "content-security-policy" not in {k.lower() for k in headers}
