import { describe, expect, it, beforeEach } from "vitest";
import {
  hfOnboardingSkipKey,
  isHfOnboardingSkipped,
  needsHfTokenOnboarding,
  skipHfOnboarding,
} from "./hfOnboarding";
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

describe("hfOnboarding", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it("shows onboarding when no valid token and not skipped", () => {
    expect(needsHfTokenOnboarding(baseStatus(), "user-1")).toBe(true);
  });

  it("hides onboarding when token is valid", () => {
    const status = baseStatus();
    status.connectivity.token_valid = true;
    expect(needsHfTokenOnboarding(status, "user-1")).toBe(false);
  });

  it("hides onboarding after skip", () => {
    skipHfOnboarding("user-1");
    expect(isHfOnboardingSkipped("user-1")).toBe(true);
    expect(needsHfTokenOnboarding(baseStatus(), "user-1")).toBe(false);
  });

  it("uses per-user skip keys", () => {
    skipHfOnboarding("user-a");
    expect(hfOnboardingSkipKey("user-b")).not.toBe(hfOnboardingSkipKey("user-a"));
    expect(needsHfTokenOnboarding(baseStatus(), "user-b")).toBe(true);
  });
});