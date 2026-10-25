"""Three-way matching: purchase order  vs  goods receipt (GRN)  vs  supplier invoice.

All money and quantities use Decimal. The engine only REPORTS discrepancies; it
never changes a document. Inputs use the same dict shape the extractor produces.
"""
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher

SIMILAR_NAME = 0.85          # item names this alike count as the same item
SIMILAR_SUPPLIER = 0.80
SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class Tolerances:
    """How much difference is accepted before it is reported."""
    price_pct: Decimal = Decimal("1")        # unit price may differ by this many percent
    qty_abs: Decimal = Decimal("0")          # quantity difference allowed (units)
    amount_abs: Decimal = Decimal("0.01")    # rounding allowance on money amounts


def _d(value) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def normalize_name(text) -> str:
    """Lower-case, letters and digits only, single spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def aggregate_items(items: list[dict]) -> list[dict]:
    """Merge lines with the same name: quantities add up; a price is kept only if all lines agree."""
    merged: dict[str, dict] = {}
    for item in items or []:
        key = normalize_name(item.get("name"))
        if not key:
            continue
        qty, price, line = _d(item.get("quantity")), _d(item.get("unit_price")), _d(item.get("line_total"))
        if key not in merged:
            merged[key] = {"key": key, "name": item.get("name"), "quantity": qty or Decimal(0),
                           "prices": [price] if price is not None else [],
                           "line_total": line, "price_conflict": False}
        else:
            entry = merged[key]
            entry["quantity"] += qty or Decimal(0)
            if price is not None:
                entry["prices"].append(price)
            if line is not None:
                entry["line_total"] = (entry["line_total"] or Decimal(0)) + line
    for entry in merged.values():
        prices = set(entry.pop("prices"))
        entry["price_conflict"] = len(prices) > 1          # same item listed at different prices
        entry["unit_price"] = next(iter(prices)) if len(prices) == 1 else None
    return list(merged.values())


def pair_items(source: list[dict], target: list[dict]) -> dict[int, int | None]:
    """Map each source item index to a target index (or None).

    Exact normalised names first, then clearly-similar names. If two targets are
    about equally similar we do NOT guess; the item stays unmatched.
    """
    result: dict[int, int | None] = {i: None for i in range(len(source))}
    used: set[int] = set()
    by_key = {t["key"]: j for j, t in enumerate(target)}
    for i, s in enumerate(source):
        j = by_key.get(s["key"])
        if j is not None and j not in used:
            result[i] = j
            used.add(j)
    for i, s in enumerate(source):
        if result[i] is not None:
            continue
        scores = sorted(((_similar(s["key"], t["key"]), j) for j, t in enumerate(target) if j not in used),
                        reverse=True)
        if scores and scores[0][0] >= SIMILAR_NAME and (len(scores) == 1 or scores[0][0] - scores[1][0] > 0.03):
            result[i] = scores[0][1]
            used.add(scores[0][1])
    return result


def _issue(code, severity, message, item=None, expected=None, actual=None) -> dict:
    return {"code": code, "severity": severity, "item": item, "message": message,
            "expected": None if expected is None else str(expected),
            "actual": None if actual is None else str(actual)}


def _fmt(value: Decimal | None) -> str:
    return "n/a" if value is None else str(value)


# --------------------------------------------------------------- duplicates
def check_duplicates(invoice: dict, others: list[dict]) -> list[dict]:
    """Compare against invoices saved earlier (lower id), or all others when ids are unknown."""
    issues = []
    my_id = invoice.get("id")
    number = normalize_name(invoice.get("document_id"))
    vendor = normalize_name(invoice.get("vendor"))
    total = _d(invoice.get("total"))
    for other in others or []:
        other_id = other.get("id")
        if my_id is not None and other_id is not None and other_id >= my_id:
            continue
        same_vendor = vendor and _similar(vendor, normalize_name(other.get("vendor"))) >= SIMILAR_SUPPLIER
        if invoice.get("file_sha256") and invoice.get("file_sha256") == other.get("file_sha256"):
            issues.append(_issue("DUPLICATE_INVOICE", "high",
                                 f"This is the same file as an invoice already saved (record {other_id})."))
        elif same_vendor and number and number == normalize_name(other.get("document_id")):
            issues.append(_issue("DUPLICATE_INVOICE", "high",
                                 f"Invoice number {invoice.get('document_id')} from this supplier was already "
                                 f"submitted (record {other_id}).", actual=invoice.get("document_id")))
        elif (same_vendor and total is not None and total == _d(other.get("total"))
              and invoice.get("document_date") and invoice.get("document_date") == other.get("document_date")):
            issues.append(_issue("POSSIBLE_DUPLICATE", "medium",
                                 f"Same supplier, date and total as invoice {other.get('document_id')} "
                                 f"(record {other_id}) but a different number.", actual=_fmt(total)))
    return issues


# -------------------------------------------------------------------- match
def three_way_match(po: dict | None, grns: list[dict], invoice: dict,
                    other_invoices: list[dict] | None = None,
                    tol: Tolerances = Tolerances()) -> dict:
    """Return {status, discrepancies, items, supported_subtotal, notes}."""
    issues: list[dict] = []
    rows: list[dict] = []
    notes: list[str] = []
    grns = grns or []

    if po is None:
        issues.append(_issue("PO_NOT_FOUND", "high",
                             f"No purchase order found for reference {invoice.get('po_reference') or '(none)'}."))
    if not grns:
        issues.append(_issue("GRN_NOT_FOUND", "high",
                             "No goods receipt recorded for this order. Goods may not have arrived."))

    # ---- header checks
    if po is not None:
        a, b = normalize_name(po.get("vendor")), normalize_name(invoice.get("vendor"))
        if a and b and _similar(a, b) < SIMILAR_SUPPLIER:
            issues.append(_issue("SUPPLIER_MISMATCH", "high", "Invoice supplier differs from the PO supplier.",
                                 expected=po.get("vendor"), actual=invoice.get("vendor")))
        if po.get("currency") and invoice.get("currency") and po["currency"] != invoice["currency"]:
            issues.append(_issue("CURRENCY_MISMATCH", "high", "Invoice currency differs from the PO currency.",
                                 expected=po["currency"], actual=invoice["currency"]))
        if po.get("document_date") and invoice.get("document_date") and invoice["document_date"] < po["document_date"]:
            issues.append(_issue("INVOICE_BEFORE_PO", "medium", "Invoice is dated before the purchase order.",
                                 expected=po["document_date"], actual=invoice["document_date"]))

    # ---- item checks
    inv_items = aggregate_items(invoice.get("items"))
    po_items = aggregate_items(po.get("items")) if po else []
    grn_items = aggregate_items([i for g in grns for i in (g.get("items") or [])])
    to_po = pair_items(inv_items, po_items) if po else {i: None for i in range(len(inv_items))}
    to_grn = pair_items(inv_items, grn_items)
    supported = Decimal(0)
    supported_known = po is not None
    seen_po: set[int] = set()
    seen_grn: set[int] = set()

    for i, inv in enumerate(inv_items):
        name = inv["name"]
        row = {"item": name, "po_qty": None, "received_qty": None, "invoiced_qty": inv["quantity"],
               "po_price": None, "invoice_price": inv["unit_price"], "result": "OK"}
        problems = []
        pj, gj = to_po[i], to_grn[i]
        po_item = po_items[pj] if pj is not None else None
        grn_item = grn_items[gj] if gj is not None else None
        if pj is not None:
            seen_po.add(pj)
        if gj is not None:
            seen_grn.add(gj)

        if po is not None and po_item is None:
            issues.append(_issue("ITEM_NOT_ON_PO", "high", f"“{name}” is invoiced but is not on the purchase order.",
                                 item=name, actual=_fmt(inv["quantity"])))
            problems.append("not on PO")
            supported_known = False
        if po_item is not None:
            row["po_qty"], row["po_price"] = po_item["quantity"], po_item["unit_price"]
            if inv["quantity"] > po_item["quantity"] + tol.qty_abs:
                issues.append(_issue("QTY_EXCEEDS_PO", "high",
                                     f"“{name}”: invoiced quantity is more than ordered.", item=name,
                                     expected=_fmt(po_item["quantity"]), actual=_fmt(inv["quantity"])))
                problems.append("qty > PO")
            pp, ip = po_item["unit_price"], inv["unit_price"]
            if pp is not None and ip is not None:
                allowed = max(pp * tol.price_pct / Decimal(100), tol.amount_abs)
                if ip - pp > allowed:
                    issues.append(_issue("PRICE_HIGHER_THAN_PO", "high",
                                         f"“{name}”: unit price is higher than the PO price.", item=name,
                                         expected=_fmt(pp), actual=_fmt(ip)))
                    problems.append("price high")
                elif pp - ip > allowed:
                    issues.append(_issue("PRICE_LOWER_THAN_PO", "medium",
                                         f"“{name}”: unit price is lower than the PO price.", item=name,
                                         expected=_fmt(pp), actual=_fmt(ip)))
                    problems.append("price low")
            elif pp is None and ip is not None:
                notes.append(f"“{name}”: the PO has no single unit price for this item, so price was not compared.")

        if grns:
            received = grn_item["quantity"] if grn_item else Decimal(0)
            row["received_qty"] = received
            if grn_item is None:
                issues.append(_issue("ITEM_NOT_RECEIVED", "high",
                                     f"“{name}” is invoiced but no receipt records it.", item=name,
                                     expected="0", actual=_fmt(inv["quantity"])))
                problems.append("not received")
            elif inv["quantity"] > received + tol.qty_abs:
                issues.append(_issue("QTY_EXCEEDS_RECEIPT", "high",
                                     f"“{name}”: invoiced quantity is more than was received.", item=name,
                                     expected=_fmt(received), actual=_fmt(inv["quantity"])))
                problems.append("qty > received")
            elif received - inv["quantity"] > tol.qty_abs:
                issues.append(_issue("QTY_BELOW_RECEIPT", "low",
                                     f"“{name}”: invoice covers less than was received (a later invoice may follow).",
                                     item=name, expected=_fmt(received), actual=_fmt(inv["quantity"])))
                problems.append("under-billed")

        qty, price, line = inv["quantity"], inv["unit_price"], inv["line_total"]
        if price is not None and line is not None and abs(qty * price - line) > tol.amount_abs:
            issues.append(_issue("INVOICE_LINE_ARITHMETIC", "medium",
                                 f"“{name}”: quantity × price does not equal the line total.", item=name,
                                 expected=_fmt(qty * price), actual=_fmt(line)))
            problems.append("line total wrong")

        # amount supported by both PO and receipt (what could safely be paid for this line)
        if po_item is not None and po_item["unit_price"] is not None:
            limit = min(inv["quantity"], po_item["quantity"],
                        row["received_qty"] if grns and row["received_qty"] is not None else inv["quantity"])
            supported += max(limit, Decimal(0)) * po_item["unit_price"]
        else:
            supported_known = False
        row["result"] = ", ".join(problems) if problems else "OK"
        rows.append(row)

    # PO / GRN lines that the invoice does not mention
    for j, p in enumerate(po_items):
        if j not in seen_po:
            rows.append({"item": p["name"], "po_qty": p["quantity"], "received_qty": None, "invoiced_qty": None,
                         "po_price": p["unit_price"], "invoice_price": None, "result": "not invoiced yet"})
    if po:
        po_keys = pair_items(grn_items, po_items)
        for k, g in enumerate(grn_items):
            pj = po_keys[k]
            if pj is None:
                issues.append(_issue("GRN_ITEM_NOT_ON_PO", "medium",
                                     f"“{g['name']}” was received but is not on the purchase order.",
                                     item=g["name"], actual=_fmt(g["quantity"])))
            elif g["quantity"] > po_items[pj]["quantity"] + tol.qty_abs:
                issues.append(_issue("OVER_DELIVERY", "medium",
                                     f"“{g['name']}”: more was received than ordered.", item=g["name"],
                                     expected=_fmt(po_items[pj]["quantity"]), actual=_fmt(g["quantity"])))

    # ---- totals
    subtotal, tax, total = _d(invoice.get("subtotal")), _d(invoice.get("tax")), _d(invoice.get("total"))
    line_sum = sum((i["line_total"] for i in inv_items if i["line_total"] is not None), Decimal(0))
    if subtotal is not None and inv_items and all(i["line_total"] is not None for i in inv_items):
        if abs(line_sum - subtotal) > tol.amount_abs:
            issues.append(_issue("INVOICE_SUBTOTAL_MISMATCH", "medium",
                                 "Invoice line totals do not add up to the subtotal.",
                                 expected=_fmt(line_sum), actual=_fmt(subtotal)))
    if None not in (subtotal, tax, total) and abs(subtotal + tax - total) > tol.amount_abs:
        issues.append(_issue("INVOICE_TOTAL_MISMATCH", "medium", "Subtotal + tax does not equal the invoice total.",
                             expected=_fmt(subtotal + tax), actual=_fmt(total)))
    if po is not None and _d(po.get("total")) is not None and total is not None \
            and total > _d(po["total"]) + tol.amount_abs:
        issues.append(_issue("TOTAL_EXCEEDS_PO", "medium", "Invoice total is higher than the purchase order total.",
                             expected=_fmt(_d(po["total"])), actual=_fmt(total)))

    issues.extend(check_duplicates(invoice, other_invoices or []))

    worst = max((SEVERITY_RANK[i["severity"]] for i in issues), default=0)
    status = "Hold - review needed" if worst >= 2 else "Matched with minor differences" if worst == 1 else "Matched"
    return {"status": status, "discrepancies": issues, "items": rows,
            "supported_subtotal": str(supported) if supported_known and inv_items else None, "notes": notes}
