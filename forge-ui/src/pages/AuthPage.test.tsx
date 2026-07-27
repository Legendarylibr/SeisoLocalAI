import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthPage } from "./AuthPage";

const confirmKeyBackup = vi.fn();
const downloadNip49KeyBackup = vi.fn().mockResolvedValue(undefined);

vi.mock("@/hooks/useAuth", () => ({
  useAuth: () => ({
    needsOnboarding: false,
    storageModeConfigured: true,
    keyBackup: {
      nsec: "nsec1backupsecretvalue",
      npub: "npub1publicidentityvalue",
    },
    login: vi.fn(),
    register: vi.fn(),
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

  it("shows the generated nsec as the write-down secret and npub as identity", () => {
    render(<AuthPage />);

    expect(screen.getByRole("heading", { name: /write down your nsec/i })).toBeTruthy();
    expect(screen.getByLabelText("Your nsec").textContent).toContain("nsec1backupsecretvalue");
    expect(screen.getByLabelText("Your npub").textContent).toContain("npub1publicidentityvalue");
    expect(screen.queryByText(/write this npub down now/i)).toBeNull();
    expect(screen.getByText(/write this nsec down now/i)).toBeTruthy();
  });

  it("downloads a NIP-49 encrypted backup after passphrase entry", async () => {
    render(<AuthPage />);

    fireEvent.change(screen.getByLabelText(/backup passphrase/i), {
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
