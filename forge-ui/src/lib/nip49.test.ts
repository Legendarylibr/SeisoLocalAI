import { describe, expect, it } from "vitest";
import { bech32 } from "@scure/base";
import {
  decryptNip49,
  encryptNip49,
  looksLikeNcryptsec,
  resolveSecretToNsec,
  secretBytesToNsec,
} from "./nip49";

const BECH32_LIMIT = 5000;
const VECTOR_NCRYPTSEC =
  "ncryptsec1qgg9947rlpvqu76pj5ecreduf9jxhselq2nae2kghhvd5g7dgjtcxfqtd67p9m0w57lspw8gsq6yphnm8623nsl8xn9j4jdzz84zm3frztj3z7s35vpzmqf6ksu8r89qk5z2zxfmu5gv8th8wclt0h4p";
const VECTOR_SECRET_HEX =
  "3501454135014541350145413501453fefb02227e449e57cf4d3a3ce05378683";

function mutateLogN(ncryptsec: string, logn: number): string {
  const { prefix, words } = bech32.decode(ncryptsec as `${string}1${string}`, BECH32_LIMIT);
  expect(prefix).toBe("ncryptsec");
  const bytes = new Uint8Array(bech32.fromWords(words));
  bytes[1] = logn & 0xff;
  return bech32.encode("ncryptsec", bech32.toWords(bytes), BECH32_LIMIT);
}

describe("nip49", () => {
  it("decrypts the official NIP-49 test vector", () => {
    const secret = decryptNip49(VECTOR_NCRYPTSEC, "nostr");
    const hex = Array.from(secret, (b) => b.toString(16).padStart(2, "0")).join("");
    expect(hex).toBe(VECTOR_SECRET_HEX);
  });

  it("round-trips encrypt/decrypt", () => {
    const secret = decryptNip49(VECTOR_NCRYPTSEC, "nostr");
    const enc = encryptNip49(secret, "another-pass", 16, 0x00);
    expect(enc.startsWith("ncryptsec1")).toBe(true);
    expect(decryptNip49(enc, "another-pass")).toEqual(secret);
  });

  it("resolves nsec and ncryptsec (+ file paste) to nsec", async () => {
    const secret = decryptNip49(VECTOR_NCRYPTSEC, "nostr");
    const nsec = secretBytesToNsec(secret);
    expect(await resolveSecretToNsec(nsec)).toBe(nsec);
    expect(await resolveSecretToNsec(VECTOR_NCRYPTSEC, "nostr")).toBe(nsec);
    expect(
      await resolveSecretToNsec(`ncryptsec=${VECTOR_NCRYPTSEC}\nnpub=npub1x`, "nostr"),
    ).toBe(nsec);
  });

  it("detects ncryptsec paste shapes", () => {
    expect(looksLikeNcryptsec(VECTOR_NCRYPTSEC)).toBe(true);
    expect(looksLikeNcryptsec(`ncryptsec=${VECTOR_NCRYPTSEC}`)).toBe(true);
    expect(looksLikeNcryptsec("nsec1qqqq")).toBe(false);
  });

  it("rejects log_n outside 1..18 on decrypt (DoS cap)", () => {
    const secret = decryptNip49(VECTOR_NCRYPTSEC, "nostr");
    const enc = encryptNip49(secret, "bound-check", 16, 0x00);
    for (const bad of [0, 19, 22, 23, 30, 255]) {
      expect(() => decryptNip49(mutateLogN(enc, bad), "bound-check")).toThrow(/log_n/);
    }
  });

  it("rejects empty and whitespace passphrases", () => {
    const secret = decryptNip49(VECTOR_NCRYPTSEC, "nostr");
    for (const bad of ["", "   ", "\t"]) {
      expect(() => encryptNip49(secret, bad)).toThrow(/passphrase/);
      expect(() => decryptNip49(VECTOR_NCRYPTSEC, bad)).toThrow(/passphrase/);
    }
  });

  it("requires passphrase for ncryptsec resolution", async () => {
    await expect(resolveSecretToNsec(VECTOR_NCRYPTSEC)).rejects.toThrow(/Passphrase/);
    await expect(resolveSecretToNsec(VECTOR_NCRYPTSEC, "wrong")).rejects.toThrow(
      /wrong passphrase|corrupted/,
    );
  });
});
