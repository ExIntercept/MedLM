"""Username + password accounts. scrypt hashing from the stdlib, no extra deps.

This is local-machine auth: it keeps two people on the same laptop out of each
other's records. It is not hardened for exposure on a public network.
"""
import hashlib
import hmac
import secrets
import time
from typing import Dict, Optional

from . import storage

SESSION_TTL_SECONDS = 60 * 60 * 12
_sessions: Dict[str, Dict] = {}

SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1


def hash_password(password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32
    )
    return {"salt": salt.hex(), "hash": digest.hex()}


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    digest = hashlib.scrypt(
        password.encode(),
        salt=bytes.fromhex(salt_hex),
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
    )
    return hmac.compare_digest(digest.hex(), hash_hex)


def register(username: str, password: str) -> str:
    if not storage.valid_username(username):
        raise ValueError("Username must be 3-32 characters: letters, numbers, . _ -")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if storage.user_exists(username):
        raise ValueError("That username is taken.")
    creds = hash_password(password)
    storage.create_user(username, {"username": username, "created": time.time(), **creds})
    return start_session(username)


def login(username: str, password: str) -> str:
    account = storage.read_account(username)
    if not account or not verify_password(password, account["salt"], account["hash"]):
        raise ValueError("Wrong username or password.")
    return start_session(username)


def start_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"username": username, "expires": time.time() + SESSION_TTL_SECONDS}
    return token


def resolve_session(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    session = _sessions.get(token)
    if not session:
        return None
    if session["expires"] < time.time():
        _sessions.pop(token, None)
        return None
    return session["username"]


def end_session(token: Optional[str]) -> None:
    if token:
        _sessions.pop(token, None)


def list_users() -> list:
    return sorted(p.name for p in storage.USERS_DIR.iterdir() if (p / "account.json").exists())
