import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthPage } from "./AuthPage";

const confirmKeyBackup = vi.fn();
const register = vi.fn();
const login = vi.fn();
const downloadNip49KeyBackup = vi.fn().mockResolvedValue(undefined);

let authMock: {
  needsOnboarding: boolean;
  storageModeConfigured: boolean;
  keyBackup: { nsec: string; npub: string } | null;
};

vi.mock("@/hooks/useAuth", () => ({
  useAuth: () => ({
    needsOnboarding: authMock.needsOnboarding,
    storageModeConfigured: authMock.storageModeConfigured,
    keyBackup: authMock.keyBackup,
    login,
    register,
    confirmKeyBackup,
    resetSession: vi.fn(),
  }),
}));

vi.mock("@/lib/keyBackup", async () => {
  const actual = await vi.importActual<typeof import("@/lib/keyBackup")>("@/lib/keyBackup");
  return {
    ...actual,
    downloadNip49KeyBackup: (...args: unknown[]) => downloadNip49KeyBackup(...args),
  };
});

vi.mock("@/components/SeisoLogo", () => ({
  SeisoLogoMark: () => <span data-testid="logo" />,
}));

vi.mock("@/components/Icons", () => ({
  IconLock: () => <span data-testid="lock" />,
}));

describe("AuthPage key backup", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("shows the generated recovery key and public ID without leading with nsec jargon", () => {
    authMock = {
      needsOnboarding: false,
      storageModeConfigured: true,
      keyBackup: {
        nsec: "nsec1backupsecretvalue",
        npub: "npub1publicidentityvalue",
      },
    };
    render(<AuthPage />);

    expect(screen.getByRole("heading", { name: /save your recovery key/i })).toBeTruthy();
    expect(screen.getByLabelText("Your recovery key").textContent).toContain("nsec1backupsecretvalue");
    expect(screen.getByLabelText("Your public ID").textContent).toContain("npub1publicidentityvalue");
    expect(screen.queryByRole("heading", { name: /write down your nsec/i })).toBeNull();
    expect(screen.getByText(/write this down or store it in a password manager now/i)).toBeTruthy();
    expect(screen.getByText(/technical names/i)).toBeTruthy();
  });

  it("downloads a NIP-49 encrypted backup after passphrase entry", async () => {
    authMock = {
      needsOnboarding: false,
      storageModeConfigured: true,
      keyBackup: {
        nsec: "nsec1backupsecretvalue",
        npub: "npub1publicidentityvalue",
      },
    };
    render(<AuthPage />);

    fireEvent.change(screen.getByLabelText(/^backup passphrase$/i), {
      target: { value: "correct-horse" },
    });
    fireEvent.change(screen.getByLabelText(/confirm passphrase/i), {
      target: { value: "correct-horse" },
    });
    fireEvent.click(screen.getByRole("button", { name: /download encrypted \.txt/i }));

    await waitFor(() => {
      expect(downloadNip49KeyBackup).toHaveBeenCalledWith(
        {
          nsec: "nsec1backupsecretvalue",
          npub: "npub1publicidentityvalue",
        },
        { passphrase: "correct-horse" },
      );
    });
  });
});

describe("AuthPage onboarding copy", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("offers create-account as the primary path and hides restore behind details", () => {
    authMock = {
      needsOnboarding: true,
      storageModeConfigured: false,
      keyBackup: null,
    };
    render(<AuthPage />);

    expect(screen.getByRole("heading", { name: /create your local account/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /create account and continue/i })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /create your nostr identity/i })).toBeNull();
    expect(screen.getByText(/already have a recovery key/i)).toBeTruthy();
    // Restore field is inside collapsed details — still in the document for a11y/import.
    expect(screen.getByLabelText(/recovery key/i)).toBeTruthy();
  });

  it("generates via the primary button without requiring a pasted key", async () => {
    authMock = {
      needsOnboarding: true,
      storageModeConfigured: true,
      keyBackup: null,
    };
    register.mockResolvedValue({ nsec: null, npub: null });
    render(<AuthPage />);

    fireEvent.click(screen.getByRole("button", { name: /create account and continue/i }));
    await waitFor(() => {
      expect(register).toHaveBeenCalledWith({ generate: true }, undefined);
    });
  });
});
