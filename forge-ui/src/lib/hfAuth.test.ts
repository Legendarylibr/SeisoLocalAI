import { describe, expect, it } from "vitest";
import { canPublishToHub } from "./hfAuth";
import type { HfHubStatus } from "@/lib/api";

const baseStatus = (): HfHubStatus => ({
  auth: {
    cli_available: false,
    cli_binary: null,
    cli_logged_in: false,
    token_configured: false,
    token_sources: [],
    user_token_saved: false,
    token_source: "none",
  },
  connectivity: {
    reachable: true,
    latency_ms: 42,
    token_valid: false,
    token_username: null,
    anonymous_ok: true,
    error: null,
  },
  transfer: {
    backend: "http",
    xet_available: false,
    xet_version: null,
    high_performance: false,
    num_threads: "4",
    download_timeout_s: "60",
    hints: [],
    hint: null,
  },
  cache_dir: "/tmp/hf",
  runtime: {
    llamacpp: true,
    mlx: false,
    torch: false,
    huggingface_hub: true,
    install_hints: [],
  },
  ready_for_download: true,
  ready_for_upload: false,
  ready_for_gguf_chat: true,
});

describe("canPublishToHub", () => {
  it("blocks upload when only public downloads are ready", () => {
    expect(canPublishToHub(baseStatus(), baseStatus().auth)).toBe(false);
  });

  it("allows upload with a valid saved token", () => {
    const status = baseStatus();
    status.connectivity.token_valid = true;
    status.ready_for_upload = true;
    expect(canPublishToHub(status, status.auth)).toBe(true);
  });

  it("allows upload with a per-request token", () => {
    expect(canPublishToHub(baseStatus(), baseStatus().auth, { requestToken: "hf_test" })).toBe(true);
  });

  it("allows upload when CLI login is preferred", () => {
    const auth = { ...baseStatus().auth, cli_logged_in: true };
    expect(canPublishToHub(baseStatus(), auth, { useCli: true })).toBe(true);
  });
});