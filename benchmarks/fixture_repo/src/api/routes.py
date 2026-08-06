"""HTTP route handlers.

A router mapping method+path to handler callables. Handlers return ``Response``
objects from the middleware module. This wires auth, users, and billing
together so cross-module retrieval has something realistic to chew on, and the
handler bodies plus validation make the file realistically sized.
"""

from __future__ import annotations

from src.auth.middleware import (
    Request,
    Response,
    auth_middleware,
    rate_limit,
    request_logger,
)
from src.billing.invoice import apply_discount, create_invoice, Discount, get_invoice
from src.billing.payment import charge
from src.users.service import create_user, deactivate_user, get_user
from src.utils.logging import get_logger

logger = get_logger(__name__)


def health(request: Request) -> Response:
    return Response(status=200, body={"status": "ok"})


def create_user_route(request: Request) -> Response:
    email = request.headers.get("X-Email", "")
    if not email:
        return Response(status=400, body={"error": "email required"})
    try:
        user = create_user(email)
    except Exception as exc:  # noqa: BLE001 - surface as 400
        return Response(status=400, body={"error": str(exc)})
    return Response(status=201, body={"id": user.id, "email": user.email})


def get_user_route(request: Request) -> Response:
    user_id = request.headers.get("X-User-Id", "")
    user = get_user(user_id)
    if user is None:
        return Response(status=404, body={"error": "not found"})
    return Response(status=200, body={"id": user.id, "email": user.email})


def delete_user_route(request: Request) -> Response:
    user_id = request.headers.get("X-User-Id", "")
    if deactivate_user(user_id):
        return Response(status=204, body={})
    return Response(status=404, body={"error": "not found"})


def create_invoice_route(request: Request) -> Response:
    user_id = request.user_id or ""
    amount = int(request.headers.get("X-Amount", "0"))
    invoice = create_invoice(user_id, amount)
    code = request.headers.get("X-Discount")
    if code:
        apply_discount(invoice.id, Discount(code=code, percent_off=10))
    return Response(status=201, body={"invoice_id": invoice.id,
                                      "total": invoice.total_cents})


def pay_invoice_route(request: Request) -> Response:
    invoice_id = request.headers.get("X-Invoice-Id", "")
    card = request.headers.get("X-Card-Token", "")
    result = charge(invoice_id, card)
    return Response(status=200, body={"status": result.status.value})


ROUTES = {
    ("GET", "/health"): health,
    ("POST", "/users"): rate_limit(create_user_route),
    ("GET", "/users"): get_user_route,
    ("DELETE", "/users"): auth_middleware(delete_user_route),
    ("POST", "/invoices"): auth_middleware(create_invoice_route),
    ("POST", "/payments"): request_logger(auth_middleware(pay_invoice_route)),
}
