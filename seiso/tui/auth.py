"""TUI Nostr auth — same owner identity, keys, and persistence as Forge."""

from __future__ import annotations

import asyncio
import hmac
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from forge.config import ForgeSettings, StorageMode, get_settings
from forge.db.store import Database
from forge.security.audit import audit_event
from forge.security.auth import (
    InvalidTokenError,
    create_access_token,
    decode_token,
    revoke_access_token,
)
from forge.services.nostr_auth import (
    NOSTR_PASSWORD_SENTINEL,
    persist_user_signing_key,
    resolve_identity,
    user_public_view,
)
from seiso.research.nostr.bech32 import bech32_encode
from seiso.research.nostr.keys import keypair_from_secret
from seiso.research.nostr.nip49 import decrypt_ncryptsec, encrypt_ncryptsec
from seiso.security import generate_secret_key

T = TypeVar("T")

DEFAULT_DISPLAY_NAME = "Admin"
SESSION_FILENAME = ".tui_session"
BACKUP_FILENAME = "seiso-ncryptsec-backup.txt"
_LOGIN_WINDOW_S = 60.0
_LOGIN_MAX = 10


class AuthError(Exception):
    """User-facing auth failure (bad nsec, rate limit, reset refused)."""


@dataclass(frozen=True, slots=True)
class AuthUser:
    id: str
    npub: str
    nostr_pubkey: str
    display_name: str

    @property
    def short_npub(self) -> str:
        if len(self.npub) <= 16:
            return self.npub
        return f"{self.npub[:12]}…{self.npub[-4:]}"


@dataclass(frozen=True, slots=True)
class AuthStatus:
    needs_onboarding: bool
    storage_mode: str
    storage_mode_configured: bool
    owner_npub: str | None
    session_valid: bool
    user: AuthUser | None


def load_settings(data_dir: Path) -> ForgeSettings:
    """Forge settings for this data dir (same files the web UI uses)."""
    resolved = Path(data_dir).expanduser().resolve()
    if get_settings.cache_info().currsize:
        cached = get_settings()
        if Path(cached.data_dir).expanduser().resolve() == resolved:
            return cached
    return ForgeSettings(data_dir=data_dir)


def session_path(data_dir: Path) -> Path:
    return Path(data_dir) / SESSION_FILENAME


def looks_like_ncryptsec(raw: str) -> bool:
    text = raw.strip().lower()
    return text.startswith("ncryptsec1") or "ncryptsec=" in text or "ncryptsec1" in text


def looks_like_nsec(raw: str) -> bool:
    return raw.strip().lower().startswith("nsec1")


def extract_ncryptsec(raw: str) -> str:
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("ncryptsec="):
            return stripped.split("=", 1)[1].strip()
        if stripped.lower().startswith("ncryptsec1"):
            return stripped
    text = raw.strip()
    if text.lower().startswith("ncryptsec1"):
        return text
    raise AuthError("No ncryptsec found in that backup")


def format_key_backup_txt(ncryptsec: str, npub: str) -> str:
    """Same backup file the Forge Auth page downloads."""
    return "\n".join(
        [
            "# Seiso Local AI — NIP-49 encrypted key backup (ncryptsec)",
            "# This file does NOT contain your raw nsec.",
            "# Decrypt with your passphrase to sign in, or paste ncryptsec + passphrase in Forge.",
            "# Keep the passphrase separate from this file.",
            "",
            f"ncryptsec={ncryptsec}",
            f"npub={npub}",
            "",
        ]
    )


def resolve_secret(raw: str, passphrase: str | None = None) -> str:
    """Turn nsec, ncryptsec, backup .txt, or hex into an nsec."""
    text = (raw or "").strip()
    if not text:
        raise AuthError("Recovery key is required")
    if "\n" not in text and len(text) < 4096:
        path = Path(text).expanduser()
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            try:
                if path.stat().st_size <= 64_000:
                    text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise AuthError(f"Cannot read {path}: {exc}") from exc
    if looks_like_ncryptsec(text):
        if not passphrase or not passphrase.strip():
            raise AuthError("Passphrase required for encrypted backup")
        try:
            secret = decrypt_ncryptsec(extract_ncryptsec(text), passphrase)
        except ValueError as exc:
            raise AuthError(str(exc)) from exc
        return bech32_encode("nsec", secret)
    try:
        return keypair_from_secret(text).nsec
    except ValueError as exc:
        raise AuthError("Invalid recovery key") from exc


def write_encrypted_backup(
    nsec: str,
    npub: str,
    passphrase: str,
    dest: Path,
) -> Path:
    if len(passphrase) < 8:
        raise AuthError("Passphrase must be at least 8 characters")
    pair = keypair_from_secret(nsec)
    # 0x00 = shown on screen during onboarding — same as Forge AuthPage.
    blob = encrypt_ncryptsec(bytes.fromhex(pair.secret_hex), passphrase, key_security=0x00)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(format_key_backup_txt(blob, npub), encoding="utf-8")
    dest.chmod(0o600)
    return dest


def user_from_row(row: dict) -> AuthUser:
    view = user_public_view(row)
    npub = str(view.get("npub") or "")
    pubkey = str(view.get("nostr_pubkey") or "")
    return AuthUser(
        id=str(view["id"]),
        npub=npub,
        nostr_pubkey=pubkey,
        display_name=str(row.get("display_name") or DEFAULT_DISPLAY_NAME),
    )


class TuiAuth:
    """Single-owner Nostr auth for the TUI, persisted under SEISO_DATA_DIR."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._attempts: list[float] = []

    def settings(self) -> ForgeSettings:
        return load_settings(self.data_dir)

    def _run(self, fn: Callable[[ForgeSettings, Database], Awaitable[T]]) -> T:
        """Run *fn* on a private event loop (TUI is sync; Forge tests already have one)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run_async(fn))
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(self._run_async(fn))).result()

    async def _run_async(self, fn: Callable[[ForgeSettings, Database], Awaitable[T]]) -> T:
        settings = self.settings()
        db = Database(
            settings.db_path,
            encryption_key=settings.db_encryption_key_bytes,
            ephemeral=bool(settings.db_ephemeral),
        )
        try:
            return await fn(settings, db)
        finally:
            await db.close()

    def _check_login_rate(self) -> None:
        now = time.monotonic()
        self._attempts = [t for t in self._attempts if t > now - _LOGIN_WINDOW_S]
        if len(self._attempts) >= _LOGIN_MAX:
            raise AuthError("Too many attempts. Wait a minute.")
        self._attempts.append(now)

    def _write_session(self, user_id: str, settings: ForgeSettings) -> None:
        if settings.db_ephemeral:
            session_path(self.data_dir).unlink(missing_ok=True)
            return
        token = create_access_token(user_id, settings)
        path = session_path(self.data_dir)
        path.write_text(token, encoding="utf-8")
        path.chmod(0o600)

    def _clear_session(self, settings: ForgeSettings | None = None) -> None:
        path = session_path(self.data_dir)
        token = ""
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            path.unlink(missing_ok=True)
        if token and settings is not None:
            revoke_access_token(token, settings)

    def read_session_token(self) -> str | None:
        path = session_path(self.data_dir)
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8").strip()
        return raw or None

    def status(self) -> AuthStatus:
        token = self.read_session_token()

        async def _inner(settings: ForgeSettings, db: Database) -> AuthStatus:
            count = await db.user_count()
            owner_npub: str | None = None
            user: AuthUser | None = None
            if count > 0:
                sole = await db.get_sole_user()
                if sole:
                    user = user_from_row(sole)
                    owner_npub = user.npub or None
            session_user: AuthUser | None = None
            if token and user is not None:
                try:
                    user_id = decode_token(token, settings)
                except InvalidTokenError:
                    user_id = ""
                if user_id and user_id == user.id:
                    session_user = user
            return AuthStatus(
                needs_onboarding=count == 0,
                storage_mode=settings.storage_mode,
                storage_mode_configured=settings.storage_mode_configured,
                owner_npub=owner_npub,
                session_valid=session_user is not None,
                user=session_user,
            )

        return self._run(_inner)

    def restore_session(self) -> AuthUser | None:
        return self.status().user

    def register(
        self,
        *,
        generate: bool = False,
        nsec: str | None = None,
        storage_mode: StorageMode | None = None,
    ) -> tuple[AuthUser, str | None]:
        self._check_login_rate()
        settings = self.settings()
        if not settings.storage_mode_configured:
            if storage_mode not in {"persistent", "ephemeral"}:
                raise AuthError("Choose persistent or ephemeral storage")
            settings.persist_storage_mode(storage_mode)
            from forge.api.deps import clear_dependency_caches

            clear_dependency_caches()
            settings = self.settings()
        try:
            identity = resolve_identity(nsec=nsec, generate=generate)
        except ValueError as exc:
            raise AuthError(str(exc)) from exc

        async def _inner(cur: ForgeSettings, db: Database) -> dict:
            try:
                return await db.create_first_user(
                    NOSTR_PASSWORD_SENTINEL,
                    DEFAULT_DISPLAY_NAME,
                    nostr_pubkey=identity.pubkey_hex,
                )
            except ValueError as exc:
                raise AuthError(str(exc)) from exc

        row = self._run(_inner)
        persist_user_signing_key(
            data_dir=settings.data_dir,
            user_id=row["id"],
            pair=identity.pair,
            persist=not settings.db_ephemeral,
        )
        settings.sync_inference_api_key_owner(identity.pubkey_hex)
        self._write_session(row["id"], settings)
        audit_event(
            "auth_register",
            user_id=row["id"],
            nostr_pubkey=identity.pubkey_hex,
            generated=generate,
        )
        user = user_from_row({**row, "nostr_pubkey": identity.pubkey_hex})
        return user, (identity.nsec if generate else None)

    def login(self, nsec: str) -> AuthUser:
        self._check_login_rate()
        try:
            identity = resolve_identity(nsec=nsec, generate=False)
        except ValueError as exc:
            audit_event("auth_login_failed")
            raise AuthError("Invalid credentials") from exc

        async def _inner(settings: ForgeSettings, db: Database) -> tuple[dict, ForgeSettings]:
            user = await db.get_sole_user()
            stored = str((user or {}).get("nostr_pubkey") or "").strip().lower()
            presented = identity.pubkey_hex.strip().lower()
            if (
                not user
                or len(stored) != 64
                or len(presented) != 64
                or not hmac.compare_digest(stored, presented)
            ):
                audit_event("auth_login_failed")
                raise AuthError("Invalid credentials")
            return user, settings

        row, settings = self._run(_inner)
        persist_user_signing_key(
            data_dir=settings.data_dir,
            user_id=row["id"],
            pair=identity.pair,
            persist=not settings.db_ephemeral,
        )
        settings.sync_inference_api_key_owner(identity.pubkey_hex)
        self._write_session(row["id"], settings)
        audit_event("auth_login", user_id=row["id"], nostr_pubkey=identity.pubkey_hex)
        return user_from_row(row)

    def logout(self) -> None:
        settings = self.settings()
        token = self.read_session_token()
        user_id = ""
        if token:
            try:
                user_id = decode_token(token, settings)
            except InvalidTokenError:
                user_id = ""

        async def _purge(_settings: ForgeSettings, db: Database) -> None:
            if user_id:
                await db.purge_user_chat(user_id)

        if user_id:
            self._run(_purge)
        self._clear_session(settings)
        if user_id:
            audit_event("auth_logout", user_id=user_id)

    def reset_session(self, confirmation: str) -> None:
        settings = self.settings()
        if settings.allow_remote:
            raise AuthError("Session reset is only available on local-only instances")
        if confirmation.strip().upper() != "RESET":
            raise AuthError("Type RESET to confirm starting a new local session")
        if "SEISO_INFERENCE_API_KEY" in os.environ:
            raise AuthError(
                "Session reset refused: SEISO_INFERENCE_API_KEY is env-bound. Unset it before wipe."
            )

        async def _inner(cur: ForgeSettings, db: Database) -> None:
            await db.reset_local_session()

        self._run(_inner)
        from forge.services.nostr_settings import wipe_nostr_identity_material

        wipe_nostr_identity_material(settings.data_dir)
        if "SEISO_SECRET_KEY" not in os.environ:
            key_file = settings.data_dir / ".secret_key"
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text(generate_secret_key(), encoding="utf-8")
            key_file.chmod(0o600)
        settings.clear_inference_api_key_owner()
        if not settings.rotate_inference_api_key():
            raise AuthError("Session reset refused: Compat /v1 key could not be rotated")
        self._clear_session(settings)
        audit_event("auth_reset_session", owner_cleared=True, nostr_identity_wiped=True)
        from forge.api.deps import clear_dependency_caches, close_dependency_caches

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(close_dependency_caches())
        clear_dependency_caches()

    def rotate_key(self, user_id: str) -> tuple[str, str]:
        from forge.services.nostr_settings import generate_user_nostr_key

        settings = self.settings()
        result = generate_user_nostr_key(
            settings.data_dir,
            user_id,
            persist=not settings.db_ephemeral,
        )

        async def _inner(_settings: ForgeSettings, db: Database) -> None:
            await db.update_user_nostr_pubkey(user_id, result["pubkey_hex"])

        self._run(_inner)
        settings.sync_inference_api_key_owner(result["pubkey_hex"])
        self._write_session(user_id, settings)
        return result["nsec"], result["npub"]

    def import_key(self, user_id: str, secret: str) -> str:
        from forge.services.nostr_settings import import_user_nostr_key

        settings = self.settings()
        try:
            result = import_user_nostr_key(
                settings.data_dir,
                user_id,
                secret,
                persist=not settings.db_ephemeral,
            )
        except ValueError as exc:
            raise AuthError(str(exc)) from exc

        async def _inner(_settings: ForgeSettings, db: Database) -> None:
            await db.update_user_nostr_pubkey(user_id, result["pubkey_hex"])

        self._run(_inner)
        settings.sync_inference_api_key_owner(result["pubkey_hex"])
        self._write_session(user_id, settings)
        return result["npub"]

    def nostr_status(self, user_id: str, auth_pubkey: str | None) -> dict:
        from forge.services.nostr_settings import nostr_status

        settings = self.settings()
        return nostr_status(
            settings.data_dir,
            user_id,
            auth_pubkey=auth_pubkey,
            persist_keys=not settings.db_ephemeral,
        )

    def save_prefs(
        self,
        user_id: str,
        *,
        auto_attest: bool,
        relays: list[str],
        allow_loopback: bool,
    ) -> dict:
        from forge.services.nostr_settings import NostrPrefs, save_nostr_prefs
        from seiso.security import SecurityError

        settings = self.settings()
        try:
            return save_nostr_prefs(
                settings.data_dir,
                user_id,
                NostrPrefs(
                    auto_attest=auto_attest,
                    relays=relays,
                    allow_loopback=allow_loopback,
                ),
            ).to_dict()
        except SecurityError as exc:
            raise AuthError(str(exc)) from exc
