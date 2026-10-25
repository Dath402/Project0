import json

import pymupdf
import pytest

from src import pipeline
from src.ocr_reader import OCRError, OCRUnavailableError, boxes_to_text

SAMPLE_LINES = [
    "INVOICE", "Invoice No: INV-002", "Vendor: Test Traders", "Invoice Date: 20/10/2026", "Currency: INR",
]


def make_pdf(lines=None, table=True, password=None) -> bytes:
    """Synthetic digital PDF; table cells are placed at x positions (real columns)."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    for text in lines or SAMPLE_LINES:
        page.insert_text((50, y), text, fontsize=11)
        y += 18
    if table:
        y += 10
        for row in [("Item Description", "Qty", "Unit Price", "Amount"),
                    ("Steel Bolt", "100", "2.50", "250.00"),
                    ("Hex Nut", "200", "1,000.00", "200,000.00")]:
            for x, cell in zip((50, 250, 340, 450), row):
                page.insert_text((x, y), cell, fontsize=11)
            y += 18
        y += 10
        for text in ("Subtotal: 200250.00", "GST: 36045.00", "Grand Total: 236295.00"):
            page.insert_text((50, y), text, fontsize=11)
            y += 18
    if password:
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password)
    return doc.tobytes()


def test_digital_pdf_end_to_end():
    result = pipeline.process_document(make_pdf(), "inv.pdf", "Invoice")
    data = result["data"]
    assert result["text_source"].startswith("PyMuPDF")
    assert data["document_id"] == "INV-002" and data["vendor"] == "Test Traders"
    assert data["document_date"] == "2026-10-20"
    assert [i["name"] for i in data["items"]] == ["Steel Bolt", "Hex Nut"]
    assert data["items"][1]["unit_price"] == "1000.00" and data["items"][1]["line_total"] == "200000.00"
    assert data["total"] == "236295.00" and data["source_filename"] == "inv.pdf"
    assert json.loads(json.dumps(data)) == data


def test_empty_file():
    result = pipeline.process_document(b"", "a.pdf", "Invoice")
    assert any("empty" in w.lower() for w in result["warnings"])
    assert result["data"]["items"] == []


def test_corrupt_pdf():
    result = pipeline.process_document(b"%PDF-1.4 this is garbage", "a.pdf", "Invoice")
    assert any("corrupt" in w.lower() for w in result["warnings"])


def test_wrong_content_for_extension():
    result = pipeline.process_document(b"MZ not a pdf", "a.pdf", "Invoice")
    assert any("does not match" in w for w in result["warnings"])


def test_unsupported_type():
    result = pipeline.process_document(b"hello", "notes.txt", "Invoice")
    assert any("Unsupported" in w for w in result["warnings"])


def test_too_large(monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_FILE_BYTES", 10)
    result = pipeline.process_document(b"%PDF-" + b"x" * 50, "a.pdf", "Invoice")
    assert any("larger" in w for w in result["warnings"])


def test_encrypted_pdf():
    result = pipeline.process_document(make_pdf(password="secret"), "locked.pdf", "Invoice")
    assert any("password" in w.lower() for w in result["warnings"])


def test_scanned_pdf_without_ocr_installed(monkeypatch):
    def fail(*a, **k):
        raise OCRUnavailableError("missing")
    monkeypatch.setattr(pipeline, "ocr_pdf_bytes", fail)
    blank = pymupdf.open()
    blank.new_page()
    result = pipeline.process_document(blank.tobytes(), "scan.pdf", "Invoice")
    assert any("requirements-ocr.txt" in w for w in result["warnings"])
    assert result["data"]["items"] == []


def test_ocr_runtime_failure_does_not_crash(monkeypatch):
    def fail(*a, **k):
        raise OCRError("model download failed")
    monkeypatch.setattr(pipeline, "ocr_image_bytes", fail)
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
    result = pipeline.process_document(png, "scan.png", "Invoice")
    assert any("model download failed" in w for w in result["warnings"])


def test_unexpected_ocr_exception_does_not_crash(monkeypatch):
    def boom(*a, **k):
        raise ValueError("weird")
    monkeypatch.setattr(pipeline, "ocr_image_bytes", boom)
    result = pipeline.process_document(b"\xff\xd8\xff" + b"0" * 20, "p.jpg", "Invoice")
    assert any("OCR failed" in w for w in result["warnings"])


def test_image_uses_ocr_text(monkeypatch):
    monkeypatch.setattr(pipeline, "ocr_image_bytes",
                        lambda b, s: "Invoice No: IMG-5\nVendor: Pic Co\nTotal: 10.00")
    result = pipeline.process_document(b"\x89PNG\r\n\x1a\n" + b"0" * 20, "p.png", "Invoice")
    assert result["data"]["document_id"] == "IMG-5"
    assert result["text_source"] == "PaddleOCR (image)"


def test_boxes_to_text_rebuilds_rows():
    texts = ["Pen", "2", "10.00", "Total"]
    boxes = [(10, 100, 50, 120), (200, 101, 210, 121), (300, 99, 350, 119), (10, 150, 60, 170)]
    lines = boxes_to_text(texts, boxes).splitlines()
    assert lines[0].split() == ["Pen", "2", "10.00"] and "   " in lines[0]
    assert lines[1] == "Total"
    assert boxes_to_text([], []) == ""
