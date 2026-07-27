import { afterEach, describe, expect, it } from "vitest";
import {
  KEY_BACKUP_STORAGE,
  parseKeyBackup,
  persistKeyBackup,
  readStoredKeyBackup,
} from "./keyBackup";

describe("keyBackup", () => {
  afterEach(() => {
    sessionStorage.clear();
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
});
