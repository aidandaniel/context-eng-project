"""Product catalog.

Completely unrelated to auth/users/billing. Pure noise to give the benchmark a
larger surface area, so naive full-file context gathering pays a real cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Product:
    sku: str
    name: str
    price_cents: int
    tags: tuple[str, ...] = ()


_CATALOG: dict[str, Product] = {}


def add_product(product: Product) -> None:
    _CATALOG[product.sku] = product


def get_product(sku: str) -> Product | None:
    return _CATALOG.get(sku)


def search_products(term: str) -> list[Product]:
    term = term.lower()
    return [
        p
        for p in _CATALOG.values()
        if term in p.name.lower() or any(term in t for t in p.tags)
    ]


def all_products() -> list[Product]:
    return list(_CATALOG.values())
