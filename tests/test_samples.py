"""The sample PDFs in sample_docs/ should survive the whole chain: read, save, match."""
from pathlib import Path

import pytest

from src import db
from src.matching import three_way_match
from src.pipeline import process_document

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"
TYPES = {"PO-5001": "Purchase Order", "GRN-6001": "Delivery Receipt / GRN",
         "INV-7001": "Invoice", "INV-7002": "Invoice"}


@pytest.fixture(scope="module")
def results():
    return {n: process_document((DOCS / f"{n}.pdf").read_bytes(), f"{n}.pdf", t) for n, t in TYPES.items()}


def test_all_samples_are_read_from_the_pdf_text(results):
    for r in results.values():
        assert r["text_source"].startswith("PyMuPDF")
        assert len(r["data"]["items"]) == 3
        assert r["file_sha256"]


def test_extracted_fields(results):
    po = results["PO-5001"]["data"]
    assert (po["document_id"], po["delivery_date"], po["total"]) == ("PO-5001", "2026-10-30", "29500.00")
    assert po["items"][1] == {"name": "Pressure Gauge", "quantity": "10", "unit_price": "1250.00", "line_total": "12500.00"}
    grn = results["GRN-6001"]["data"]
    assert grn["po_reference"] == "PO-5001" and grn["items"][1]["quantity"] == "8"
    assert results["INV-7002"]["data"]["po_reference"] == "PO-5001"


def test_save_then_match_matches_expected_outcome(tmp_path, results):
    with db.get_conn(str(tmp_path / "s.db")) as c:
        for r in results.values():
            db.save_document(c, r["data"], r["file_sha256"])
        po, grns = db.get_po(c, "PO-5001"), db.get_grns_for_po(c, "PO-5001")
        out = {i["document_id"]: three_way_match(po, grns, i, db.all_invoice_headers(c)) for i in db.list_invoices(c)}
    assert out["INV-7001"]["status"] == "Matched"
    bad = {d["code"] for d in out["INV-7002"]["discrepancies"]}
    assert {"PRICE_HIGHER_THAN_PO", "QTY_EXCEEDS_RECEIPT"} <= bad
