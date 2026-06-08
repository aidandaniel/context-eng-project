"""Payment processing stubs.

Simulates a payment gateway with retries and idempotency keys. Unrelated to
authentication; included as realistic noise for the benchmark.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

from src.billing.invoice import get_invoice, mark_paid
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PaymentStatus(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class PaymentResult:
    invoice_id: str
    status: PaymentStatus
    message: str = ""
    transaction_id: str = ""


@dataclass
class PaymentAttempt:
    invoice_id: str
    idempotency_key: str
    attempts: int = 0
    history: list[PaymentStatus] = field(default_factory=list)


_ATTEMPTS: dict[str, PaymentAttempt] = {}
_MAX_ATTEMPTS = 3


def _attempt_for(invoice_id: str, key: str) -> PaymentAttempt:
    attempt = _ATTEMPTS.get(key)
    if attempt is None:
        attempt = PaymentAttempt(invoice_id=invoice_id, idempotency_key=key)
        _ATTEMPTS[key] = attempt
    return attempt


def charge(invoice_id: str, card_token: str, idempotency_key: str | None = None) -> PaymentResult:
    """Charge an invoice against a card token, with retry accounting."""
    key = idempotency_key or uuid.uuid4().hex
    attempt = _attempt_for(invoice_id, key)
    attempt.attempts += 1

    invoice = get_invoice(invoice_id)
    if invoice is None:
        result = PaymentResult(invoice_id, PaymentStatus.FAILED, "no such invoice")
    elif not card_token:
        result = PaymentResult(invoice_id, PaymentStatus.FAILED, "missing card")
    elif attempt.attempts > _MAX_ATTEMPTS:
        result = PaymentResult(invoice_id, PaymentStatus.FAILED, "too many attempts")
    else:
        mark_paid(invoice_id)
        result = PaymentResult(
            invoice_id, PaymentStatus.SUCCEEDED, transaction_id=uuid.uuid4().hex
        )
        logger.info("charged invoice %s", invoice_id)

    attempt.history.append(result.status)
    return result


def refund(invoice_id: str) -> PaymentResult:
    invoice = get_invoice(invoice_id)
    if invoice is None or not invoice.paid:
        return PaymentResult(invoice_id, PaymentStatus.FAILED, "not refundable")
    return PaymentResult(invoice_id, PaymentStatus.REFUNDED)
