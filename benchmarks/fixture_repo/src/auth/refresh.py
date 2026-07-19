"""Session token refresh and logout handling.

This module owns the lifecycle of refresh tokens: issuing a new access token
from a valid refresh token, rotating refresh tokens, revoking sessions, and
clearing session state on logout. It is the most security-sensitive part of the
auth package, so the logic here is intentionally explicit rather than clever.

The public surface is small (``refreshToken``, ``rotate_refresh_token``,
``logout``) but there is a fair amount of supporting machinery for revocation
lists, grace periods, and audit logging that a maintainer would scroll past
while hunting for the one function relevant to their task -- which is exactly
the cost the context bundle avoids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.auth.tokens import decode_claims, make_token, verify_token
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

REFRESH_TTL_SECONDS = 60 * 60 * 24 * 14
ACCESS_TTL_SECONDS = 60 * 15
GRACE_PERIOD_SECONDS = 30


class RefreshError(Exception):
    """Raised when a refresh token is invalid, expired, or already used."""


class RevokedError(RefreshError):
    """Raised when a refresh token has been explicitly revoked."""


@dataclass
class Session:
    user_id: str
    refresh_token: str | None
    issued_at: float
    last_seen: float = 0.0
    device: str = "unknown"
    metadata: dict = field(default_factory=dict)


# In-memory revocation list. A real system would back this with Redis or a DB.
_REVOKED: set[str] = set()


def _is_revoked(token: str) -> bool:
    return token in _REVOKED


def revoke(token: str) -> None:
    """Add a refresh token to the revocation list."""
    _REVOKED.add(token)
    logger.info("revoked refresh token")


def refreshToken(session: Session) -> str:
    """Issue a new access token for an active session.

    Raises RefreshError if the session has no refresh token (e.g. the user has
    already logged out) which previously surfaced as a confusing TypeError when
    ``verify_token`` was handed ``None``. The guard below is the actual fix.
    """
    if session.refresh_token is None:
        raise RefreshError("session has no refresh token; user logged out")

    if _is_revoked(session.refresh_token):
        raise RevokedError("refresh token has been revoked")

    claims = verify_token(session.refresh_token)
    if claims is None:
        raise RefreshError("refresh token failed verification")

    age = time.time() - session.issued_at
    if age > REFRESH_TTL_SECONDS + GRACE_PERIOD_SECONDS:
        raise RefreshError("refresh token expired")

    session.last_seen = time.time()
    logger.info("refreshing token for user %s", session.user_id)
    return make_token(session.user_id, kind="access")


def rotate_refresh_token(session: Session) -> str:
    """Issue and store a brand-new refresh token, invalidating the old one."""
    if session.refresh_token is not None:
        revoke(session.refresh_token)
    new_token = make_token(session.user_id, kind="refresh")
    session.refresh_token = new_token
    session.issued_at = time.time()
    session.last_seen = session.issued_at
    return new_token


def introspect(session: Session) -> dict:
    """Return a diagnostic view of the session's token state."""
    if session.refresh_token is None:
        return {"active": False, "reason": "logged_out"}
    claims = decode_claims(session.refresh_token)
    return {
        "active": not _is_revoked(session.refresh_token),
        "user_id": session.user_id,
        "issued_at": session.issued_at,
        "age_seconds": time.time() - session.issued_at,
        "claims": claims,
    }


def time_until_expiry(session: Session) -> float:
    """Seconds remaining before the refresh token expires (may be negative)."""
    elapsed = time.time() - session.issued_at
    return REFRESH_TTL_SECONDS - elapsed


def logout(session: Session) -> None:
    """Clear all token state for a session.

    After logout the session must not be usable for ``refreshToken``; that call
    will raise RefreshError rather than passing None into verification.
    """
    logger.info("logging out user %s", session.user_id)
    if session.refresh_token is not None:
        revoke(session.refresh_token)
    session.refresh_token = None
    session.issued_at = 0.0
    session.last_seen = 0.0


def logout_all_devices(sessions: list[Session]) -> int:
    """Revoke and clear a batch of sessions for a single user."""
    count = 0
    for session in sessions:
        if session.refresh_token is not None:
            logout(session)
            count += 1
    return count
