"""Request authentication middleware.

Wraps incoming requests, extracts the bearer token, verifies it, and attaches
the resolved user id to the request context. On a missing or invalid token it
short-circuits with a 401 without touching downstream handlers.

Besides the core ``auth_middleware`` there are several composable middlewares
(rate limiting, request logging, CORS) so the file is realistically sized and a
reader benefits from being pointed at just the relevant function.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from src.auth.refresh import RefreshError, Session, refreshToken
from src.auth.tokens import verify_token
from src.utils.logging import get_logger

logger = get_logger(__name__)

Handler = Callable[["Request"], "Response"]


@dataclass
class Request:
    method: str = "GET"
    path: str = "/"
    headers: dict[str, str] = field(default_factory=dict)
    user_id: str | None = None
    received_at: float = field(default_factory=time.time)


@dataclass
class Response:
    status: int
    body: dict
    headers: dict[str, str] = field(default_factory=dict)


def _extract_bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip()


def auth_middleware(handler: Handler) -> Handler:
    """Return a wrapped handler that enforces authentication.

    A previous bug returned 401 for valid tokens because the bearer prefix was
    matched case-sensitively; ``_extract_bearer`` now lower-cases the prefix.
    """

    def wrapped(request: Request) -> Response:
        token = _extract_bearer(request)
        if token is None:
            return Response(status=401, body={"error": "missing token"})
        claims = verify_token(token)
        if claims is None:
            return Response(status=401, body={"error": "invalid token"})
        request.user_id = claims.get("sub")
        return handler(request)

    return wrapped


_RATE_BUCKETS: dict[str, list[float]] = defaultdict(list)


def rate_limit(handler: Handler, limit: int = 60, window: float = 60.0) -> Handler:
    """Simple fixed-window rate limiter keyed by user id or path."""

    def wrapped(request: Request) -> Response:
        key = request.user_id or request.path
        now = time.time()
        bucket = [t for t in _RATE_BUCKETS[key] if now - t < window]
        bucket.append(now)
        _RATE_BUCKETS[key] = bucket
        if len(bucket) > limit:
            return Response(status=429, body={"error": "rate limited"})
        return handler(request)

    return wrapped


def request_logger(handler: Handler) -> Handler:
    def wrapped(request: Request) -> Response:
        response = handler(request)
        logger.info(
            "%s %s -> %d (%dms)",
            request.method,
            request.path,
            response.status,
            int((time.time() - request.received_at) * 1000),
        )
        return response

    return wrapped


def with_cors(handler: Handler, origin: str = "*") -> Handler:
    def wrapped(request: Request) -> Response:
        response = handler(request)
        response.headers["Access-Control-Allow-Origin"] = origin
        return response

    return wrapped


def refresh_if_needed(session: Session) -> str | None:
    """Best-effort token refresh used by long-lived connections."""
    try:
        return refreshToken(session)
    except RefreshError as exc:
        logger.warning("refresh failed: %s", exc)
        return None
