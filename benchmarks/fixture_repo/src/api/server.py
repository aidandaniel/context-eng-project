"""Minimal request dispatcher tying routes together.

Resolves a method+path to a handler, builds the request object, invokes the
handler inside a top-level error boundary, and records basic metrics. Kept
small but realistic; ``dispatch`` is the function most queries about routing
care about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.api.routes import ROUTES
from src.auth.middleware import Request, Response
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Metrics:
    total: int = 0
    by_status: dict[int, int] = field(default_factory=dict)

    def record(self, status: int) -> None:
        self.total += 1
        self.by_status[status] = self.by_status.get(status, 0) + 1


_METRICS = Metrics()


def dispatch(method: str, path: str, headers: dict[str, str] | None = None) -> Response:
    """Route a request to its handler, or 404 if unmatched."""
    handler = ROUTES.get((method, path))
    request = Request(method=method, path=path, headers=headers or {})
    if handler is None:
        response = Response(status=404, body={"error": "no route"})
    else:
        try:
            response = handler(request)
        except Exception as exc:  # noqa: BLE001 - top-level boundary
            logger.error("unhandled error for %s %s: %s", method, path, exc)
            response = Response(status=500, body={"error": "internal"})
    _METRICS.record(response.status)
    return response


def metrics_snapshot() -> dict:
    return {"total": _METRICS.total, "by_status": dict(_METRICS.by_status)}


def run() -> None:
    logger.info("fixture server ready with %d routes", len(ROUTES))
    started = time.time()
    logger.info("startup at %.0f", started)
