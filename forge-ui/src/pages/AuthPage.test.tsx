import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthPage } from "./AuthPage";

const confirmKeyBackup = vi.fn();
const downloadKeyBackupTxt = vi.fn();

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
    downloadKeyBackupTxt: (...args: unknown[]) => downloadKeyBackupTxt(...args),
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

  it("offers a same-window .txt download of the key backup", () => {
    render(<AuthPage />);

    fireEvent.click(screen.getByRole("button", { name: /download \.txt/i }));
    expect(downloadKeyBackupTxt).toHaveBeenCalledWith({
      nsec: "nsec1backupsecretvalue",
      npub: "npub1publicidentityvalue",
    });
  });
});
