/**
 * NIP-49 private key encryption (ncryptsec).
 * @see https://github.com/nostr-protocol/nips/blob/master/49.md
 */
import { bech32 } from "@scure/base";
import { scrypt } from "@noble/hashes/scrypt";
import { xchacha20poly1305 } from "@noble/ciphers/chacha";
import { concatBytes, randomBytes } from "@noble/hashes/utils";

const BECH32_LIMIT = 5000;
const PAYLOAD_LEN = 91;

export type KeySecurityByte = 0x00 | 0x01 | 0x02;

/** Decode nsec1… → 32-byte secret. */
export function nsecToSecretBytes(nsec: string): Uint8Array {
  const raw = nsec.trim();
  const { prefix, words } = bech32.decode(raw as `${string}1${string}`, BECH32_LIMIT);
  if (prefix !== "nsec") {
    throw new Error(`invalid prefix ${prefix}, expected nsec`);
  }
  const bytes = new Uint8Array(bech32.fromWords(words));
  if (bytes.length !== 32) {
    throw new Error("invalid nsec length");
  }
  return bytes;
}

/** Encode 32-byte secret → nsec1… */
export function secretBytesToNsec(secret: Uint8Array): string {
  if (secret.length !== 32) {
    throw new Error("secret must be 32 bytes");
  }
  return bech32.encode("nsec", bech32.toWords(secret), BECH32_LIMIT);
}

export function encryptNip49(
  secret: Uint8Array,
  password: string,
  logn: number = 16,
  keySecurity: KeySecurityByte = 0x02,
): string {
  if (secret.length !== 32) {
    throw new Error("secret must be 32 bytes");
  }
  if (logn < 1 || logn > 22) {
    throw new Error("log_n must be between 1 and 22");
  }
  if (!password || !password.trim()) {
    throw new Error("passphrase is required");
  }
  const salt = randomBytes(16);
  const key = scrypt(password.normalize("NFKC"), salt, {
    N: 2 ** logn,
    r: 8,
    p: 1,
    dkLen: 32,
  });
  const nonce = randomBytes(24);
  const aad = Uint8Array.from([keySecurity]);
  const ciphertext = xchacha20poly1305(key, nonce, aad).encrypt(secret);
  const payload = concatBytes(
    Uint8Array.from([0x02]),
    Uint8Array.from([logn]),
    salt,
    nonce,
    aad,
    ciphertext,
  );
  if (payload.length !== PAYLOAD_LEN) {
    throw new Error(`unexpected ncryptsec payload length ${payload.length}`);
  }
  return bech32.encode("ncryptsec", bech32.toWords(payload), BECH32_LIMIT);
}

export function decryptNip49(ncryptsec: string, password: string): Uint8Array {
  const raw = ncryptsec.trim();
  if (!password) {
    throw new Error("passphrase is required");
  }
  const { prefix, words } = bech32.decode(raw as `${string}1${string}`, BECH32_LIMIT);
  if (prefix !== "ncryptsec") {
    throw new Error(`invalid prefix ${prefix}, expected ncryptsec`);
  }
  const b = new Uint8Array(bech32.fromWords(words));
  if (b.length !== PAYLOAD_LEN) {
    throw new Error("invalid ncryptsec payload length");
  }
  if (b[0] !== 0x02) {
    throw new Error(`invalid ncryptsec version ${b[0]}`);
  }
  const logn = b[1]!;
  if (logn < 1 || logn > 22) {
    throw new Error("log_n must be between 1 and 22");
  }
  if (!password.trim()) {
    throw new Error("passphrase is required");
  }
  const salt = b.slice(2, 18);
  const nonce = b.slice(18, 42);
  const ksb = b[42]!;
  const ciphertext = b.slice(43);
  const key = scrypt(password.normalize("NFKC"), salt, {
    N: 2 ** logn,
    r: 8,
    p: 1,
    dkLen: 32,
  });
  try {
    return xchacha20poly1305(key, nonce, Uint8Array.from([ksb])).decrypt(ciphertext);
  } catch {
    throw new Error("wrong passphrase or corrupted ncryptsec");
  }
}

/** Resolve nsec1… or ncryptsec1… (+ passphrase) to an nsec for Forge auth APIs. */
export async function resolveSecretToNsec(
  secret: string,
  passphrase?: string,
): Promise<string> {
  const raw = secret.trim();
  if (raw.startsWith("nsec1")) {
    // Validate shape.
    nsecToSecretBytes(raw);
    return raw;
  }
  if (raw.startsWith("ncryptsec1")) {
    if (!passphrase) {
      throw new Error("Passphrase required to decrypt ncryptsec");
    }
    const bytes = decryptNip49(raw, passphrase);
    return secretBytesToNsec(bytes);
  }
  // Allow pasting backup file contents: ncryptsec=...
  const match = raw.match(/ncryptsec\s*=\s*(ncryptsec1[a-z0-9]+)/i);
  if (match?.[1]) {
    if (!passphrase) {
      throw new Error("Passphrase required to decrypt ncryptsec");
    }
    return secretBytesToNsec(decryptNip49(match[1], passphrase));
  }
  throw new Error("Paste an nsec1… or ncryptsec1… key");
}

export function looksLikeNcryptsec(secret: string): boolean {
  const raw = secret.trim();
  return raw.startsWith("ncryptsec1") || /ncryptsec\s*=\s*ncryptsec1/i.test(raw);
}
