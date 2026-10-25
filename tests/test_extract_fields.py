import json
from decimal import Decimal

from src.extract_fields import (SCHEMA_KEYS, extract_fields, extract_items,
                                parse_amount, parse_date)

PIPE_TEXT = """INVOICE
Invoice No: INV-001
Vendor: ABC Supplies
Invoice Date: 2026-10-20
Currency: INR

Item Description | Quantity | Unit Price | Amount
Wireless Mouse | 10 | 500.00 | 5000.00
USB Keyboard | 5 | 750.00 | 3750.00

Subtotal: 8750.00
GST: 1575.00
Grand Total: 10325.00
"""

WS_TEXT = """Item Description     Qty     Unit Price     Amount
Blue Pen (Box of 10)     2     1,250.50     2,501.00
Notebook A4     12     45.00     540.00
Subtotal     3,041.00
"""


def test_pipe_rows():
    items, warnings = extract_items(PIPE_TEXT)
    assert [i["name"] for i in items] == ["Wireless Mouse", "USB Keyboard"]
    assert items[0] == {"name": "Wireless Mouse", "quantity": "10",
                        "unit_price": "500.00", "line_total": "5000.00"}
    assert warnings == []


def test_whitespace_rows_keep_column_gaps():
    """Regression test: the 2+ space gaps must survive until the row parser."""
    items, warnings = extract_items(WS_TEXT)
    assert len(items) == 2
    assert items[0]["name"] == "Blue Pen (Box of 10)"
    assert items[0]["quantity"] == "2"
    assert items[0]["unit_price"] == "1250.50"       # thousands separator handled
    assert items[0]["line_total"] == "2501.00"
    assert items[1]["name"] == "Notebook A4"
    assert warnings == []


def test_header_and_summary_lines_are_not_items():
    items, _ = extract_items(WS_TEXT)
    names = " ".join(i["name"].lower() for i in items)
    assert "description" not in names and "subtotal" not in names


def test_aligned_numbers_without_header_are_not_guessed():
    text = "Flat 12   Green Park   110016   4\n"
    items, _ = extract_items(text)
    assert items == []


def test_ambiguous_extra_columns_not_guessed():
    text = "Item | Qty | Unit Price | Tax | Amount\nPen | 2 | 10.00 | 1.80 | 21.80\n"
    items, warnings = extract_items(text)
    assert items == []
    assert any("Pen" in w for w in warnings)


def test_amount_parsing():
    assert parse_amount("1,234.50") == Decimal("1234.50")
    assert parse_amount("₹ 1,23,456.00") == Decimal("123456.00")
    assert parse_amount("Rs. 500") == Decimal("500")
    assert parse_amount("1.234,56") is None         # ambiguous European format
    assert parse_amount("abc") is None
    assert parse_amount(None) is None


def test_date_formats():
    assert parse_date("2026-10-20") == ("2026-10-20", False)
    assert parse_date("20/10/2026") == ("2026-10-20", False)
    assert parse_date("20 Oct 2026") == ("2026-10-20", False)
    assert parse_date("October 20, 2026") == ("2026-10-20", False)
    assert parse_date("03/04/2026") == ("2026-04-03", True)   # day/month, flagged ambiguous
    assert parse_date("31/02/2026") == (None, False)          # impossible date
    assert parse_date("no date here") == (None, False)


def test_full_extraction():
    data, _ = extract_fields(PIPE_TEXT, "Invoice")
    assert data["document_id"] == "INV-001"
    assert data["vendor"] == "ABC Supplies"
    assert data["document_date"] == "2026-10-20"
    assert data["currency"] == "INR"
    assert (data["subtotal"], data["tax"], data["total"]) == ("8750.00", "1575.00", "10325.00")


def test_subtotal_not_mistaken_for_total():
    data, _ = extract_fields("Subtotal: 100.00\nTax: 18.00\nTotal: 118.00\n", "Invoice")
    assert data["subtotal"] == "100.00" and data["total"] == "118.00"


def test_total_qty_and_previous_balance_ignored():
    text = "Total Qty: 15\nPrevious Balance: 900.00\nSubtotal: 100.00\nGrand Total: 100.00\n"
    data, _ = extract_fields(text, "Invoice")
    assert data["total"] == "100.00"


def test_amount_due_warns():
    data, warnings = extract_fields("Subtotal: 100.00\nAmount Due: 600.00\n", "Invoice")
    assert data["total"] == "600.00"
    assert any("previous balance" in w for w in warnings)


def test_split_gst():
    data, warnings = extract_fields("CGST 9%: 90.00\nSGST 9%: 90.00\nTotal: 1180.00\n", "Invoice")
    assert data["tax"] == "180.00"
    assert any("CGST" in w for w in warnings)


def test_missing_vendor_and_id_are_none():
    data, _ = extract_fields("Subtotal: 10.00\nTotal: 10.00\n", "Invoice")
    assert data["document_id"] is None and data["vendor"] is None
    assert data["document_date"] is None and data["currency"] is None


def test_document_ids_by_type():
    assert extract_fields("PO Number: PO-77\nTotal: 1.00", "Purchase Order")[0]["document_id"] == "PO-77"
    assert extract_fields("GRN No: GRN-9\nTotal: 1.00", "Delivery Receipt / GRN")[0]["document_id"] == "GRN-9"


def test_conflicting_currencies_left_empty():
    data, warnings = extract_fields("Price $5\nTotal ₹400.00\n", "Invoice")
    assert data["currency"] is None
    assert any("currency" in w.lower() for w in warnings)


def test_empty_text():
    data, warnings = extract_fields("   \n", "Invoice")
    assert set(data) == set(SCHEMA_KEYS)
    assert data["items"] == [] and warnings


def test_json_serializable_and_consistent_schema():
    for doc_type in ("Invoice", "Purchase Order", "Delivery Receipt / GRN"):
        data, _ = extract_fields(PIPE_TEXT, doc_type)
        assert set(data) == set(SCHEMA_KEYS)
        assert json.loads(json.dumps(data)) == data


# ---- fields added for matching ----
def test_po_reference_on_invoice_and_grn():
    inv, _ = extract_fields("Invoice No: INV-9\nPO Number: PO-1001\nTotal: 5.00\n", "Invoice")
    assert inv["document_id"] == "INV-9" and inv["po_reference"] == "PO-1001"
    grn, _ = extract_fields("GRN No: GRN-7\nPO No: PO-1001\n", "Delivery Receipt / GRN")
    assert grn["document_id"] == "GRN-7" and grn["po_reference"] == "PO-1001"


def test_po_has_no_po_reference_of_its_own():
    po, _ = extract_fields("PURCHASE ORDER\nPO Number: PO-1001\nTotal: 5.00\n", "Purchase Order")
    assert po["document_id"] == "PO-1001" and po["po_reference"] is None


def test_delivery_date_found():
    po, _ = extract_fields("PO Number: PO-1\nOrder Date: 2026-10-10\nDelivery Date: 30/10/2026\n", "Purchase Order")
    assert po["document_date"] == "2026-10-10" and po["delivery_date"] == "2026-10-30"


def test_grn_quantity_only_rows():
    text = "GRN No: GRN-1\nItem Description | Qty Received\nBrass Valve | 20\nPressure Gauge | 8\n"
    data, _ = extract_fields(text, "Delivery Receipt / GRN")
    assert [(i["name"], i["quantity"]) for i in data["items"]] == [("Brass Valve", "20"), ("Pressure Gauge", "8")]
    assert data["items"][0]["unit_price"] is None


def test_grn_quantity_only_aligned_rows():
    text = "Item Description     Qty Received\nBrass Valve     20\nPressure Gauge     8\nReceived by: Stores\n"
    items, _ = extract_items(text, allow_quantity_only=True)
    assert [(i["name"], i["quantity"]) for i in items] == [("Brass Valve", "20"), ("Pressure Gauge", "8")]


def test_quantity_only_rows_not_accepted_for_invoices():
    items, _ = extract_items("Item Description | Qty\nBrass Valve | 20\n")
    assert items == []
