import { afterEach, describe, expect, it, vi } from "vitest";
import {
  KEY_BACKUP_FILENAME,
  KEY_BACKUP_STORAGE,
  downloadNip49KeyBackup,
  formatKeyBackupTxt,
  parseKeyBackup,
  persistKeyBackup,
  readStoredKeyBackup,
} from "./keyBackup";
import { decryptNip49, nsecToSecretBytes, secretBytesToNsec } from "./nip49";

const VECTOR_SECRET_HEX =
  "3501454135014541350145413501453fefb02227e449e57cf4d3a3ce05378683";

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

  it("formats a NIP-49 backup without raw nsec", () => {
    const text = formatKeyBackupTxt("ncryptsec1abc", "npub1public");
    expect(text).toContain("ncryptsec=ncryptsec1abc");
    expect(text).toContain("npub=npub1public");
    expect(text).not.toContain("nsec=");
    expect(text).toMatch(/NIP-49/i);
  });

  it("downloads a NIP-49 encrypted .txt in the same window", async () => {
    const secret = Uint8Array.from(
      VECTOR_SECRET_HEX.match(/.{2}/g)!.map((b) => parseInt(b, 16)),
    );
    const backup = {
      nsec: secretBytesToNsec(secret),
      npub: "npub1publicidentityvalue",
    };

    const click = vi.fn();
    const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    let blobText = "";
    vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
      // jsdom Blob may not support text(); read via FileReader sync mock
      void (blob as Blob).text?.().then((t) => {
        blobText = t;
      });
      return "blob:key-backup";
    });
    // Capture blob at construction time
    const OriginalBlob = globalThis.Blob;
    globalThis.Blob = class extends OriginalBlob {
      constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
        super(parts, options);
        blobText = String(parts?.[0] ?? "");
      }
    } as typeof Blob;

    vi.spyOn(document.body, "appendChild").mockImplementation((node) => node);
    const remove = vi.fn();
    const createElement = vi.spyOn(document, "createElement").mockReturnValue({
      href: "",
      download: "",
      rel: "",
      click,
      remove,
    } as unknown as HTMLAnchorElement);

    await downloadNip49KeyBackup(backup, { passphrase: "test-pass-123" });

    expect(createElement).toHaveBeenCalledWith("a");
    expect(click).toHaveBeenCalled();
    expect(remove).toHaveBeenCalled();
    expect(revoke).toHaveBeenCalledWith("blob:key-backup");
    const anchor = createElement.mock.results[0]?.value as HTMLAnchorElement;
    expect(anchor.download).toBe(KEY_BACKUP_FILENAME);
    expect(blobText).toContain("ncryptsec=ncryptsec1");
    expect(blobText).not.toContain(backup.nsec);
    expect(blobText).not.toContain("nsec=");

    const match = blobText.match(/ncryptsec=(ncryptsec1[a-z0-9]+)/i);
    expect(match?.[1]).toBeTruthy();
    const decrypted = decryptNip49(match![1]!, "test-pass-123");
    const toHex = (bytes: Uint8Array) =>
      Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
    expect(toHex(decrypted)).toBe(toHex(nsecToSecretBytes(backup.nsec)));

    globalThis.Blob = OriginalBlob;
  });
});
