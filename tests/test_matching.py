import copy
from decimal import Decimal

from src.matching import Tolerances, normalize_name, pair_items, aggregate_items, three_way_match


def item(name, qty, price=None, line=None):
    return {"name": name, "quantity": str(qty), "unit_price": None if price is None else str(price),
            "line_total": None if line is None else str(line)}


def po(items=None, **kw):
    base = {"document_type": "Purchase Order", "document_id": "PO-1", "vendor": "ABC Supplies",
            "document_date": "2026-10-01", "currency": "INR", "total": None,
            "items": items or [item("Wireless Mouse", 10, 100, 1000), item("USB Keyboard", 5, 200, 1000)]}
    base.update(kw)
    return base


def grn(items=None, number="G1"):
    return {"document_id": number, "items": items or [item("Wireless Mouse", 10), item("USB Keyboard", 5)]}


def inv(items=None, **kw):
    base = {"document_type": "Invoice", "document_id": "I-1", "po_reference": "PO-1", "vendor": "ABC Supplies",
            "document_date": "2026-10-10", "currency": "INR", "subtotal": "2000.00", "tax": "0.00", "total": "2000.00",
            "items": items or [item("Wireless Mouse", 10, 100, 1000), item("USB Keyboard", 5, 200, 1000)]}
    base.update(kw)
    return base


def codes(result):
    return sorted(d["code"] for d in result["discrepancies"])


def test_clean_match():
    r = three_way_match(po(), [grn()], inv())
    assert r["status"] == "Matched" and r["discrepancies"] == []
    assert r["supported_subtotal"] == "2000"


def test_price_higher_and_lower_than_po():
    high = three_way_match(po(), [grn()], inv([item("Wireless Mouse", 10, 120, 1200), item("USB Keyboard", 5, 200, 1000)],
                                              subtotal="2200.00", total="2200.00"))
    assert "PRICE_HIGHER_THAN_PO" in codes(high) and high["status"].startswith("Hold")
    low = three_way_match(po(), [grn()], inv([item("Wireless Mouse", 10, 80, 800), item("USB Keyboard", 5, 200, 1000)],
                                             subtotal="1800.00", total="1800.00"))
    assert "PRICE_LOWER_THAN_PO" in codes(low)


def test_price_within_tolerance_is_accepted():
    r = three_way_match(po(), [grn()], inv([item("Wireless Mouse", 10, "100.50", "1005.00"), item("USB Keyboard", 5, 200, 1000)],
                                           subtotal="2005.00", total="2005.00"))
    assert codes(r) == []
    strict = three_way_match(po(), [grn()], inv([item("Wireless Mouse", 10, "100.50", "1005.00"), item("USB Keyboard", 5, 200, 1000)],
                                                subtotal="2005.00", total="2005.00"), tol=Tolerances(price_pct=Decimal("0")))
    assert "PRICE_HIGHER_THAN_PO" in codes(strict)


def test_quantity_exceeds_receipt_and_po():
    r = three_way_match(po(), [grn([item("Wireless Mouse", 8), item("USB Keyboard", 5)])], inv())
    assert "QTY_EXCEEDS_RECEIPT" in codes(r)
    r2 = three_way_match(po(), [grn([item("Wireless Mouse", 12), item("USB Keyboard", 5)])],
                         inv([item("Wireless Mouse", 12, 100, 1200), item("USB Keyboard", 5, 200, 1000)],
                             subtotal="2200.00", total="2200.00"))
    assert "QTY_EXCEEDS_PO" in codes(r2) and "OVER_DELIVERY" in codes(r2)


def test_quantity_tolerance():
    r = three_way_match(po(), [grn([item("Wireless Mouse", 9), item("USB Keyboard", 5)])], inv(),
                        tol=Tolerances(qty_abs=Decimal("1")))
    assert "QTY_EXCEEDS_RECEIPT" not in codes(r)


def test_under_billing_is_only_low_severity():
    r = three_way_match(po(), [grn()], inv([item("Wireless Mouse", 6, 100, 600), item("USB Keyboard", 5, 200, 1000)],
                                           subtotal="1600.00", total="1600.00"))
    assert codes(r) == ["QTY_BELOW_RECEIPT"] and r["status"] == "Matched with minor differences"


def test_item_not_on_po_and_not_received():
    r = three_way_match(po(), [grn()], inv([item("Wireless Mouse", 10, 100, 1000),
                                                          item("USB Keyboard", 5, 200, 1000),
                                                          item("Safety Gloves", 2, 50, 100)],
                                           subtotal="2100.00", total="2100.00"))
    assert {"ITEM_NOT_ON_PO", "ITEM_NOT_RECEIVED"} <= set(codes(r))


def test_missing_grn_and_missing_po():
    assert "GRN_NOT_FOUND" in codes(three_way_match(po(), [], inv()))
    r = three_way_match(None, [grn()], inv())
    assert "PO_NOT_FOUND" in codes(r) and r["status"].startswith("Hold")


def test_header_mismatches():
    r = three_way_match(po(), [grn()], inv(vendor="Totally Different Traders", currency="USD", document_date="2026-09-01"))
    assert {"SUPPLIER_MISMATCH", "CURRENCY_MISMATCH", "INVOICE_BEFORE_PO"} <= set(codes(r))
    ok = three_way_match(po(), [grn()], inv(vendor="ABC Supplies Ltd."))
    assert "SUPPLIER_MISMATCH" not in codes(ok)


def test_arithmetic_and_totals():
    bad_line = three_way_match(po(), [grn()], inv([item("Wireless Mouse", 10, 100, 900), item("USB Keyboard", 5, 200, 1000)],
                                                  subtotal="1900.00", total="1900.00"))
    assert "INVOICE_LINE_ARITHMETIC" in codes(bad_line)
    bad_sub = three_way_match(po(), [grn()], inv(subtotal="2500.00", total="2500.00"))
    assert "INVOICE_SUBTOTAL_MISMATCH" in codes(bad_sub)
    bad_total = three_way_match(po(), [grn()], inv(tax="100.00", total="2000.00"))
    assert "INVOICE_TOTAL_MISMATCH" in codes(bad_total)
    over_po = three_way_match(po(total="2000.00"), [grn()], inv(tax="360.00", total="2360.00"))
    assert "TOTAL_EXCEEDS_PO" in codes(over_po)


def test_duplicate_by_number_later_one_only():
    first = inv(id=1)
    second = inv(id=2)
    headers = [{"id": 1, "document_id": "I-1", "vendor": "ABC Supplies", "document_date": "2026-10-10", "total": "2000.00"},
               {"id": 2, "document_id": "i-1", "vendor": "ABC Supplies", "document_date": "2026-10-10", "total": "2000.00"}]
    assert "DUPLICATE_INVOICE" not in codes(three_way_match(po(), [grn()], first, headers))
    assert "DUPLICATE_INVOICE" in codes(three_way_match(po(), [grn()], second, headers))


def test_possible_duplicate_same_total_and_date_and_same_file():
    others = [{"id": 1, "document_id": "I-OTHER", "vendor": "ABC Supplies", "document_date": "2026-10-10", "total": "2000.00"}]
    r = three_way_match(po(), [grn()], inv(id=2, document_id="I-2"), others)
    assert "POSSIBLE_DUPLICATE" in codes(r)
    same_file = [{"id": 1, "document_id": "X", "vendor": "Someone", "document_date": "2020-01-01", "total": "1", "file_sha256": "abc"}]
    r2 = three_way_match(po(), [grn()], inv(id=2, file_sha256="abc"), same_file)
    assert "DUPLICATE_INVOICE" in codes(r2)


def test_same_number_different_supplier_is_not_a_duplicate():
    others = [{"id": 1, "document_id": "I-1", "vendor": "Zeta Components",
               "document_date": "2026-10-10", "total": "5.00"}]
    assert "DUPLICATE_INVOICE" not in codes(three_way_match(po(), [grn()], inv(id=2), others))


def test_multiple_grns_are_added_up():
    g1 = grn([item("Wireless Mouse", 4), item("USB Keyboard", 5)], "G1")
    g2 = grn([item("Wireless Mouse", 6)], "G2")
    assert codes(three_way_match(po(), [g1, g2], inv())) == []


def test_partial_invoice_marks_remaining_lines():
    r = three_way_match(po(), [grn()], inv([item("Wireless Mouse", 10, 100, 1000)], subtotal="1000.00", total="1000.00"))
    assert any(row["result"] == "not invoiced yet" for row in r["items"])
    assert r["status"] == "Matched"


def test_similar_names_match_but_ambiguous_ones_do_not():
    a = aggregate_items([item("Steel Bolt M8", 1)])
    b = aggregate_items([item("Steel Bolts M8", 1)])
    assert pair_items(a, b) == {0: 0}
    ambiguous = aggregate_items([item("Bolt M8 A", 1), item("Bolt M8 B", 1)])
    assert pair_items(aggregate_items([item("Bolt M8 C", 1)]), ambiguous) == {0: None}
    assert normalize_name("  Steel-Bolt  M8! ") == "steel bolt m8"


def test_matching_does_not_modify_inputs():
    p, g, i = po(), [grn()], inv()
    before = copy.deepcopy((p, g, i))
    three_way_match(p, g, i)
    assert (p, g, i) == before


def test_decimal_precision_no_float_noise():
    r = three_way_match(po([item("Wireless Mouse", 3, "0.10", "0.30")]), [grn([item("Wireless Mouse", 3)])],
                        inv([item("Wireless Mouse", 3, "0.10", "0.30")], subtotal="0.30", tax="0.00", total="0.30"))
    assert codes(r) == []
