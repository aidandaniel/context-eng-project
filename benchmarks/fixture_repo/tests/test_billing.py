"""Tests for billing invoice + payment flow."""

from src.billing.invoice import create_invoice, get_invoice
from src.billing.payment import PaymentStatus, charge


def test_create_invoice_totals():
    invoice = create_invoice("u1", 500)
    assert invoice.total_cents == 500
    assert get_invoice(invoice.id) is invoice


def test_charge_marks_paid():
    invoice = create_invoice("u2", 1000)
    result = charge(invoice.id, card_token="tok_visa")
    assert result.status == PaymentStatus.SUCCEEDED
    assert get_invoice(invoice.id).paid is True


def test_charge_missing_card_fails():
    invoice = create_invoice("u3", 250)
    result = charge(invoice.id, card_token="")
    assert result.status == PaymentStatus.FAILED
