import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { IntegrationsPage } from "./IntegrationsPage";

const nostrKeygen = vi.fn();
const nostrStatus = vi.fn();
const listProviders = vi.fn();
const managedVllmStatus = vi.fn();
const clearNostrKey = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    listProviders: (...args: unknown[]) => listProviders(...args),
    managedVllmStatus: (...args: unknown[]) => managedVllmStatus(...args),
    nostrStatus: (...args: unknown[]) => nostrStatus(...args),
    nostrKeygen: (...args: unknown[]) => nostrKeygen(...args),
    clearNostrKey: (...args: unknown[]) => clearNostrKey(...args),
    saveNostrPrefs: vi.fn(),
    importNostrKey: vi.fn(),
    createProvider: vi.fn(),
    deleteProvider: vi.fn(),
    startManagedVllm: vi.fn(),
    stopManagedVllm: vi.fn(),
  },
}));

describe("IntegrationsPage Nostr keygen", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("shows rotated nsec in a dedicated panel, not the status toast", async () => {
    listProviders.mockResolvedValue([]);
    managedVllmStatus.mockResolvedValue(null);
    nostrStatus.mockResolvedValue({
      server_allow_nostr: true,
      key_saved: true,
      npub: "npub1old",
      identity_match: true,
      auto_attest: false,
      relays: [],
      allow_loopback: false,
    });
    nostrKeygen.mockResolvedValue({
      status: "saved",
      npub: "npub1rotatedidentityxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      nsec: "nsec1rotatedsecretvaluemustnotappearintoastxxxxxxxxxx",
    });

    render(<IntegrationsPage />);
    await waitFor(() => expect(nostrStatus).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /generate key/i }));

    await waitFor(() => expect(nostrKeygen).toHaveBeenCalled());
    const panel = await screen.findByTestId("nostr-rotated-key");
    expect(panel.textContent).toContain("nsec1rotatedsecretvaluemustnotappearintoastxxxxxxxxxx");

    const toast = screen.getByTestId("nostr-status-msg");
    expect(toast.textContent).not.toContain("nsec1");
    expect(toast.textContent).toMatch(/Save the recovery key below/i);
  });
});
