import { describe, expect, it } from "vitest";
import {
  decryptNip49,
  encryptNip49,
  looksLikeNcryptsec,
  resolveSecretToNsec,
  secretBytesToNsec,
} from "./nip49";

const VECTOR_NCRYPTSEC =
  "ncryptsec1qgg9947rlpvqu76pj5ecreduf9jxhselq2nae2kghhvd5g7dgjtcxfqtd67p9m0w57lspw8gsq6yphnm8623nsl8xn9j4jdzz84zm3frztj3z7s35vpzmqf6ksu8r89qk5z2zxfmu5gv8th8wclt0h4p";
const VECTOR_SECRET_HEX =
  "3501454135014541350145413501453fefb02227e449e57cf4d3a3ce05378683";

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
});
