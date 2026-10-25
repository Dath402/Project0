"""Checks on extracted data. Problems become warnings; source values are never changed."""
from decimal import Decimal, InvalidOperation
from datetime import date

from .extract_fields import ITEM_KEYS, SCHEMA_KEYS

TOLERANCE = Decimal("0.01")          # allowed rounding difference
MONEY_FIELDS = ("subtotal", "tax", "total")
ITEM_NUMBER_FIELDS = ("quantity", "unit_price", "line_total")
REVIEW_FIELDS = (("document_id", "Document ID"), ("vendor", "Vendor"),
                 ("document_date", "Document date"), ("total", "Total"))


def _dec(value) -> Decimal | None:
    """Strict Decimal conversion of an already-extracted string."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def validate_structure(data) -> list[str]:
    """Check keys, types and number formats."""
    if not isinstance(data, dict):
        return ["Data is not a JSON object."]
    problems = []
    for key in SCHEMA_KEYS:
        if key not in data:
            problems.append(f"Missing field “{key}”.")
    items = data.get("items")
    if not isinstance(items, list):
        problems.append("“items” must be a list.")
        items = []
    for field in MONEY_FIELDS:
        value = data.get(field)
        if value is not None and _dec(value) is None:
            problems.append(f"“{field}” is not a valid number: {value!r}.")
        elif value is not None and _dec(value) < 0:
            problems.append(f"“{field}” is negative: {value}.")
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.append(f"Item {i} is not an object.")
            continue
        for key in ITEM_KEYS:
            if key not in item:
                problems.append(f"Item {i} is missing “{key}”.")
        for field in ITEM_NUMBER_FIELDS:
            value = item.get(field)
            if value is not None and _dec(value) is None:
                problems.append(f"Item {i}: “{field}” is not a valid number: {value!r}.")
    iso = data.get("document_date")
    if iso is not None:
        try:
            date.fromisoformat(str(iso))
        except ValueError:
            problems.append(f"Date “{iso}” is not in YYYY-MM-DD format.")
    return problems


def validate_amounts(data: dict) -> list[str]:
    """Cross-check the arithmetic. Differences may be legitimate (discounts, shipping)."""
    warnings = []
    items = data.get("items") if isinstance(data.get("items"), list) else []
    line_totals = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        qty, price, line = (_dec(item.get(k)) for k in ITEM_NUMBER_FIELDS)
        name = item.get("name") or f"Item {i}"
        if None not in (qty, price, line):
            expected = qty * price
            if abs(expected - line) > TOLERANCE:
                warnings.append(f"{name}: {qty} × {price} = {expected}, but line total is {line}.")
        line_totals.append(line)

    subtotal, tax, total = (_dec(data.get(k)) for k in MONEY_FIELDS)
    if subtotal is not None and line_totals and None not in line_totals:
        items_sum = sum(line_totals, Decimal("0"))
        if abs(items_sum - subtotal) > TOLERANCE:
            warnings.append(
                f"Line totals add up to {items_sum} but subtotal is {subtotal}. "
                "This can be normal if there are discounts or other charges.")
    if None not in (subtotal, tax, total) and abs(subtotal + tax - total) > TOLERANCE:
        warnings.append(
            f"Subtotal {subtotal} + tax {tax} = {subtotal + tax}, but total is {total}. "
            "Shipping, discounts or rounding may explain this.")
    return warnings


def validate_document(data: dict) -> list[str]:
    """All checks, plus a note for each important field that is missing."""
    warnings = validate_structure(data)
    if not isinstance(data, dict):
        return warnings
    for key, label in REVIEW_FIELDS:
        if key == "total" and data.get("document_type") == "Delivery Receipt / GRN":
            continue                      # receipts normally carry no prices
        if data.get(key) is None:
            warnings.append(f"{label} was not found.")
    if data.get("document_type") in ("Invoice", "Delivery Receipt / GRN") and data.get("po_reference") is None:
        warnings.append("PO reference was not found. It is needed to link this document to its purchase order.")
    warnings.extend(validate_amounts(data))
    return warnings
