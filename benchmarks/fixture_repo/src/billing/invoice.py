"""Invoice creation, line items, discounts, and retrieval.

Independent of auth/users beyond referencing a user id. Present mainly as a
realistic, unrelated module so baseline keyword search over-fetches it while a
focused context bundle correctly ignores it. The discount handling and totals
math keep the module realistically sized.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LineItem:
    description: str
    amount_cents: int
    quantity: int = 1

    @property
    def subtotal_cents(self) -> int:
        return self.amount_cents * self.quantity


@dataclass
class Discount:
    code: str
    percent_off: int = 0
    amount_off_cents: int = 0


@dataclass
class Invoice:
    id: str
    user_id: str
    items: list[LineItem] = field(default_factory=list)
    discounts: list[Discount] = field(default_factory=list)
    paid: bool = False

    @property
    def subtotal_cents(self) -> int:
        return sum(item.subtotal_cents for item in self.items)

    @property
    def discount_cents(self) -> int:
        total = 0
        for d in self.discounts:
            total += d.amount_off_cents
            total += self.subtotal_cents * d.percent_off // 100
        return min(total, self.subtotal_cents)

    @property
    def total_cents(self) -> int:
        return self.subtotal_cents - self.discount_cents


_INVOICES: dict[str, Invoice] = {}


def create_invoice(user_id: str, amount_cents: int) -> Invoice:
    """Create an invoice with a single charge line item."""
    invoice = Invoice(id=uuid.uuid4().hex, user_id=user_id)
    invoice.items.append(LineItem(description="charge", amount_cents=amount_cents))
    _INVOICES[invoice.id] = invoice
    logger.info("created invoice %s for %s", invoice.id, user_id)
    return invoice


def add_line_item(invoice_id: str, description: str, amount_cents: int,
                  quantity: int = 1) -> bool:
    invoice = _INVOICES.get(invoice_id)
    if invoice is None:
        return False
    invoice.items.append(LineItem(description, amount_cents, quantity))
    return True


def apply_discount(invoice_id: str, discount: Discount) -> bool:
    invoice = _INVOICES.get(invoice_id)
    if invoice is None:
        return False
    invoice.discounts.append(discount)
    return True


def get_invoice(invoice_id: str) -> Invoice | None:
    return _INVOICES.get(invoice_id)


def mark_paid(invoice_id: str) -> bool:
    invoice = _INVOICES.get(invoice_id)
    if invoice is None:
        return False
    invoice.paid = True
    return True


def list_unpaid() -> list[Invoice]:
    return [i for i in _INVOICES.values() if not i.paid]
