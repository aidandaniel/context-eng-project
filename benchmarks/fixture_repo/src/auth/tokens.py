"""Low-level token creation and verification.

Tokens here are deliberately simple opaque strings rather than real JWTs so the
fixture repository has no external dependencies. The shape mirrors a real
implementation closely enough for retrieval/benchmark purposes: there is a
signing helper, a verification path, decode-without-verify for diagnostics, and
some key-rotation scaffolding that a maintainer would have to scroll past.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Optional

from src.utils.config import get_settings

# Primary and previous secrets to support zero-downtime key rotation.
_SECRET = "fixture-signing-secret"
_PREVIOUS_SECRETS = ("fixture-signing-secret-old",)

_SIGNATURE_LEN = 16


def _sign_with(secret: str, payload: bytes) -> str:
    mac = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return mac[:_SIGNATURE_LEN]


def _sign(payload: bytes) -> str:
    return _sign_with(_SECRET, payload)


def _valid_signature(payload: bytes, signature: str) -> bool:
    if hmac.compare_digest(_sign_with(_SECRET, payload), signature):
        return True
    # Accept tokens signed with a previous secret during rotation windows.
    for old in _PREVIOUS_SECRETS:
        if hmac.compare_digest(_sign_with(old, payload), signature):
            return True
    return False


def make_token(user_id: str, kind: str = "access") -> str:
    """Create a signed opaque token for a user.

    ``kind`` is one of ``access`` or ``refresh``. Adding a new ``id`` kind is a
    common feature request and is the target of one of the benchmark queries.
    """
    settings = get_settings()
    body = {
        "sub": user_id,
        "kind": kind,
        "iat": int(time.time()),
        "iss": settings.issuer,
    }
    raw = json.dumps(body, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(raw).decode()
    return f"{encoded}.{_sign(raw)}"


def _split(token: str) -> tuple[bytes, str] | None:
    try:
        encoded, signature = token.split(".", 1)
        raw = base64.urlsafe_b64decode(encoded.encode())
    except (ValueError, binascii.Error):
        return None
    return raw, signature


def verify_token(token: str) -> Optional[dict]:
    """Return the decoded claims if the token is well-formed and signed."""
    parts = _split(token)
    if parts is None:
        return None
    raw, signature = parts
    if not _valid_signature(raw, signature):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def decode_claims(token: str) -> Optional[dict]:
    """Decode claims WITHOUT verifying the signature (diagnostics only)."""
    parts = _split(token)
    if parts is None:
        return None
    raw, _ = parts
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def is_expired(token: str, ttl_seconds: int) -> bool:
    claims = decode_claims(token)
    if not claims:
        return True
    return (time.time() - claims.get("iat", 0)) > ttl_seconds


def token_kind(token: str) -> Optional[str]:
    claims = decode_claims(token)
    return claims.get("kind") if claims else None
