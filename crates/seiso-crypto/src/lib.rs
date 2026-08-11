//! AES-256-GCM field encryption compatible with `seiso.research.nostr.crypto`.
//!
//! Wire format: `enc:v1:` + base64(iv_12 || ciphertext_with_tag).
//! AAD is empty (`None` in Python cryptography AESGCM).

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use rand::RngCore;
use thiserror::Error;

/// Prefix shared with Python (`PREFIX = "enc:v1:"`).
pub const PREFIX: &str = "enc:v1:";
const IV_LEN: usize = 12;
const KEY_LEN: usize = 32;

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("encryption key is required")]
    KeyRequired,
    #[error("encryption key must be base64 or 64-char hex")]
    KeyFormat,
    #[error("encryption key must be exactly 32 bytes")]
    KeyLength,
    #[error("encryption failed")]
    Encrypt,
    #[error("decryption failed")]
    Decrypt,
    #[error("invalid ciphertext encoding")]
    Encoding,
}

/// Decode a 32-byte AES key from base64 or 64-char hex (matches Python).
pub fn resolve_encryption_key(raw: Option<&str>) -> Result<[u8; KEY_LEN], CryptoError> {
    let raw = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or(CryptoError::KeyRequired)?;
    let key = if is_hex_key(raw) {
        hex::decode(raw).map_err(|_| CryptoError::KeyFormat)?
    } else {
        B64.decode(raw.as_bytes())
            .map_err(|_| CryptoError::KeyFormat)?
    };
    if key.len() != KEY_LEN {
        return Err(CryptoError::KeyLength);
    }
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&key);
    Ok(out)
}

fn is_hex_key(raw: &str) -> bool {
    raw.len() == 64 && raw.chars().all(|c| c.is_ascii_hexdigit())
}

/// Generate a random 32-byte key.
pub fn generate_encryption_key() -> [u8; KEY_LEN] {
    let mut key = [0u8; KEY_LEN];
    rand::thread_rng().fill_bytes(&mut key);
    key
}

/// Encrypt UTF-8 plaintext; returns `enc:v1:...` string.
pub fn encrypt_field(plaintext: &str, key: &[u8; KEY_LEN]) -> Result<String, CryptoError> {
    encrypt_field_with_iv(plaintext, key, None)
}

/// Encrypt with optional fixed IV (for tests). IV must be 12 bytes when provided.
pub fn encrypt_field_with_iv(
    plaintext: &str,
    key: &[u8; KEY_LEN],
    iv: Option<[u8; IV_LEN]>,
) -> Result<String, CryptoError> {
    let mut iv_buf = [0u8; IV_LEN];
    match iv {
        Some(fixed) => iv_buf = fixed,
        None => rand::thread_rng().fill_bytes(&mut iv_buf),
    }
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| CryptoError::Encrypt)?;
    let nonce = Nonce::from_slice(&iv_buf);
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_bytes())
        .map_err(|_| CryptoError::Encrypt)?;
    let mut blob = Vec::with_capacity(IV_LEN + ciphertext.len());
    blob.extend_from_slice(&iv_buf);
    blob.extend_from_slice(&ciphertext);
    Ok(format!("{}{}", PREFIX, B64.encode(blob)))
}

/// Decrypt an `enc:v1:` value, or return the input unchanged if unprefixed
/// (matches Python `decrypt_field` plaintext passthrough).
pub fn decrypt_field(value: &str, key: &[u8; KEY_LEN]) -> Result<String, CryptoError> {
    if !value.starts_with(PREFIX) {
        return Ok(value.to_string());
    }
    let b64 = &value[PREFIX.len()..];
    let blob = B64
        .decode(b64.as_bytes())
        .map_err(|_| CryptoError::Encoding)?;
    if blob.len() < IV_LEN + 16 {
        // AES-GCM tag is 16 bytes minimum beyond IV
        return Err(CryptoError::Decrypt);
    }
    let (iv, ciphertext) = blob.split_at(IV_LEN);
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| CryptoError::Decrypt)?;
    let nonce = Nonce::from_slice(iv);
    let plain = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| CryptoError::Decrypt)?;
    String::from_utf8(plain).map_err(|_| CryptoError::Decrypt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed key bytes(range(32)) from Python vector generation.
    fn fixed_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        for (i, b) in k.iter_mut().enumerate() {
            *b = i as u8;
        }
        k
    }

    #[test]
    fn resolve_hex_and_b64() {
        let key = fixed_key();
        let hex_s = hex::encode(key);
        let b64_s = B64.encode(key);
        assert_eq!(resolve_encryption_key(Some(&hex_s)).unwrap(), key);
        assert_eq!(resolve_encryption_key(Some(&b64_s)).unwrap(), key);
    }

    #[test]
    fn roundtrip_random_iv() {
        let key = generate_encryption_key();
        let enc = encrypt_field("hello-seiso-secret", &key).unwrap();
        assert!(enc.starts_with(PREFIX));
        assert_eq!(decrypt_field(&enc, &key).unwrap(), "hello-seiso-secret");
    }

    #[test]
    fn python_vector_hello() {
        // Generated with cryptography AESGCM + fixed IV of 0x01 * 12
        let key = fixed_key();
        let enc = "enc:v1:AQEBAQEBAQEBAQEBH6aJWQeoGwO9Jwg1awIxqq4alTJBBAVxQKHRDoBOFin93w==";
        assert_eq!(decrypt_field(enc, &key).unwrap(), "hello-seiso-secret");
        // Re-encrypt with same IV must match exactly
        let iv = [1u8; 12];
        let again = encrypt_field_with_iv("hello-seiso-secret", &key, Some(iv)).unwrap();
        assert_eq!(again, enc);
    }

    #[test]
    fn python_vector_empty() {
        let key = fixed_key();
        let enc = "enc:v1:AgICAgICAgICAgICRWxrBB58GE7gJ1iWQaKtlg==";
        assert_eq!(decrypt_field(enc, &key).unwrap(), "");
        let iv = [2u8; 12];
        assert_eq!(encrypt_field_with_iv("", &key, Some(iv)).unwrap(), enc);
    }

    #[test]
    fn python_vector_unicode() {
        let key = fixed_key();
        let enc = "enc:v1:AwMDAwMDAwMDAwMDCNyZMJc+urxArZnqXcIaVY2btAQWNownM02+sWU=";
        assert_eq!(decrypt_field(enc, &key).unwrap(), "日本語🔐");
        let iv = [3u8; 12];
        assert_eq!(
            encrypt_field_with_iv("日本語🔐", &key, Some(iv)).unwrap(),
            enc
        );
    }

    #[test]
    fn plaintext_passthrough() {
        let key = fixed_key();
        assert_eq!(
            decrypt_field("not-encrypted", &key).unwrap(),
            "not-encrypted"
        );
    }
}
