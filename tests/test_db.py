import pandas as pd
import pytest

from src import db
from src.extract_fields import empty_document


def doc(kind, number, **kw):
    d = empty_document(kind)
    d.update(document_id=number, vendor="ABC", document_date="2026-10-01", currency="INR",
             items=[{"name": "Mouse", "quantity": "10", "unit_price": "100.00", "line_total": "1000.00"}],
             subtotal="1000.00", tax="180.00", total="1180.00", source_filename="x.pdf")
    d.update(kw)
    return d


def test_po_round_trip(conn):
    db.save_document(conn, doc("Purchase Order", "PO-1", delivery_date="2026-10-20"))
    po = db.get_po(conn, "PO-1")
    assert po["vendor"] == "ABC" and po["delivery_date"] == "2026-10-20"
    assert po["items"][0]["unit_price"] == "100.00" and po["total"] == "1180.00"
    assert db.get_po(conn, "nope") is None


def test_saving_a_po_again_replaces_it(conn):
    db.save_document(conn, doc("Purchase Order", "PO-1"))
    db.save_document(conn, doc("Purchase Order", "PO-1", vendor="New Vendor"))
    assert db.list_po_numbers(conn) == ["PO-1"]
    assert db.get_po(conn, "PO-1")["vendor"] == "New Vendor" and len(db.get_po(conn, "PO-1")["items"]) == 1


def test_grn_needs_po_reference(conn):
    with pytest.raises(ValueError):
        db.save_document(conn, doc("Delivery Receipt / GRN", "G1"))
    db.save_document(conn, doc("Delivery Receipt / GRN", "G1", po_reference="PO-1"))
    assert db.get_grns_for_po(conn, "PO-1")[0]["document_id"] == "G1"


def test_missing_id_and_unknown_type(conn):
    with pytest.raises(ValueError):
        db.save_document(conn, doc("Invoice", None))
    with pytest.raises(ValueError):
        db.save_document(conn, doc("Memo", "M-1"))


def test_invoices_can_be_saved_twice_so_duplicates_can_be_flagged(conn):
    db.save_document(conn, doc("Invoice", "I-1", po_reference="PO-1"), "hash1")
    db.save_document(conn, doc("Invoice", "I-1", po_reference="PO-1"), "hash1")
    assert [i["id"] for i in db.list_invoices(conn)] == [1, 2]
    assert db.get_invoice(conn, 2)["file_sha256"] == "hash1"
    assert len(db.all_invoice_headers(conn)) == 2


def test_sql_injection_text_is_stored_as_text(conn):
    db.save_document(conn, doc("Purchase Order", "PO-1", vendor="x'); DROP TABLE purchase_orders;--"))
    assert db.get_po(conn, "PO-1")["vendor"].startswith("x'); DROP")
    assert db.list_po_numbers(conn) == ["PO-1"]


def test_open_po_lines_account_for_partial_receipts(conn):
    po = doc("Purchase Order", "PO-1", delivery_date="2026-10-20")
    po["items"] = [{"name": "Mouse", "quantity": "10", "unit_price": "100", "line_total": "1000"},
                   {"name": "Keyboard", "quantity": "5", "unit_price": "200", "line_total": "1000"}]
    db.save_document(conn, po)
    g = doc("Delivery Receipt / GRN", "G1", po_reference="PO-1")
    g["items"] = [{"name": "mouse", "quantity": "4", "unit_price": None, "line_total": None},
                  {"name": "Keyboard", "quantity": "5", "unit_price": None, "line_total": None}]
    db.save_document(conn, g)
    lines = db.get_open_po_lines(conn)
    assert list(lines["material"]) == ["Mouse"] and list(lines["remaining_qty"]) == [6.0]


def test_history_and_inventory_validation(conn):
    with pytest.raises(ValueError):
        db.save_history(conn, pd.DataFrame({"supplier": ["A"]}))
    with pytest.raises(ValueError):
        db.save_inventory(conn, pd.DataFrame({"material": ["A"]}))
    inv = pd.DataFrame({"material": ["A", "A", "B"], "stock": [1, 2, 3], "daily_demand": [1, None, 1]})
    assert db.save_inventory(conn, inv) == 2
    assert len(db.load_inventory(conn)) == 2


def test_history_rows_with_missing_values_are_dropped(conn):
    df = pd.DataFrame([{"po_number": "H1", "supplier": "A", "material": "m", "order_date": "2026-01-01",
                        "quantity_ordered": 10, "unit_price": 1, "promised_lead_days": 5, "actual_lead_days": 6,
                        "quantity_received": 10},
                       {"po_number": "H2", "supplier": "A", "material": "m", "order_date": "not a date",
                        "quantity_ordered": 10, "unit_price": 1, "promised_lead_days": 5, "actual_lead_days": 6,
                        "quantity_received": 10}])
    assert db.save_history(conn, df) == 1
    assert len(db.load_history(conn)) == 1


def test_clear_all(conn):
    db.save_document(conn, doc("Purchase Order", "PO-1"))
    db.clear_all(conn)
    assert sum(db.counts(conn).values()) == 0
