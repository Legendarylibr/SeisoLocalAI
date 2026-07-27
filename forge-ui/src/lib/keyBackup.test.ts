import { afterEach, describe, expect, it, vi } from "vitest";
import {
  KEY_BACKUP_FILENAME,
  KEY_BACKUP_STORAGE,
  downloadKeyBackupTxt,
  formatKeyBackupTxt,
  parseKeyBackup,
  persistKeyBackup,
  readStoredKeyBackup,
} from "./keyBackup";

describe("keyBackup", () => {
  afterEach(() => {
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it("requires both nsec and npub bech32 prefixes", () => {
    expect(
      parseKeyBackup(JSON.stringify({ nsec: "nsec1abc", npub: "npub1xyz" })),
    ).toEqual({ nsec: "nsec1abc", npub: "npub1xyz" });
    expect(parseKeyBackup(JSON.stringify({ npub: "npub1xyz" }))).toBeNull();
    expect(parseKeyBackup(JSON.stringify({ nsec: "nsec1abc" }))).toBeNull();
    expect(
      parseKeyBackup(JSON.stringify({ nsec: "secret", npub: "npub1xyz" })),
    ).toBeNull();
  });

  it("persists and clears sessionStorage backup", () => {
    const backup = { nsec: "nsec1secret", npub: "npub1public" };
    persistKeyBackup(backup);
    expect(sessionStorage.getItem(KEY_BACKUP_STORAGE)).toContain("nsec1secret");
    expect(readStoredKeyBackup()).toEqual(backup);

    persistKeyBackup(null);
    expect(sessionStorage.getItem(KEY_BACKUP_STORAGE)).toBeNull();
    expect(readStoredKeyBackup()).toBeNull();
  });

  it("formats a plain-text backup with nsec and npub", () => {
    const text = formatKeyBackupTxt({
      nsec: "nsec1secret",
      npub: "npub1public",
    });
    expect(text).toContain("nsec=nsec1secret");
    expect(text).toContain("npub=npub1public");
    expect(text).toMatch(/^# Seiso Local AI/m);
  });

  it("downloads a .txt file in the same window", () => {
    const click = vi.fn();
    const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const createObjectURL = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:key-backup");
    vi.spyOn(document.body, "appendChild").mockImplementation((node) => node);
    const remove = vi.fn();
    const createElement = vi.spyOn(document, "createElement").mockReturnValue({
      href: "",
      download: "",
      rel: "",
      click,
      remove,
    } as unknown as HTMLAnchorElement);

    downloadKeyBackupTxt({ nsec: "nsec1secret", npub: "npub1public" });

    expect(createElement).toHaveBeenCalledWith("a");
    expect(createObjectURL).toHaveBeenCalled();
    expect(click).toHaveBeenCalled();
    expect(remove).toHaveBeenCalled();
    expect(revoke).toHaveBeenCalledWith("blob:key-backup");
    const anchor = createElement.mock.results[0]?.value as HTMLAnchorElement;
    expect(anchor.download).toBe(KEY_BACKUP_FILENAME);
    expect(anchor.href).toBe("blob:key-backup");
  });
});
