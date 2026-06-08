"""Stock level tracking. Unrelated noise module for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass


class OutOfStockError(Exception):
    pass


@dataclass
class StockLevel:
    sku: str
    quantity: int


_STOCK: dict[str, int] = {}


def set_stock(sku: str, quantity: int) -> None:
    _STOCK[sku] = max(0, quantity)


def reserve(sku: str, quantity: int) -> None:
    available = _STOCK.get(sku, 0)
    if quantity > available:
        raise OutOfStockError(sku)
    _STOCK[sku] = available - quantity


def restock(sku: str, quantity: int) -> int:
    _STOCK[sku] = _STOCK.get(sku, 0) + quantity
    return _STOCK[sku]
