"""Create synthetic sample PDFs (a PO, a GRN and two invoices) in sample_docs/.

Run:  python scripts/make_sample_pdfs.py
Everything here is made up; no real company or financial data.
"""
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "sample_docs"
COLS = (50, 270, 340, 440)


def make_pdf(path: Path, title: str, header_lines: list[str], head: tuple, rows: list[tuple], footer: list[str]) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    page.insert_text((50, y), title, fontsize=16)
    y += 30
    for line in header_lines:
        page.insert_text((50, y), line, fontsize=11)
        y += 18
    y += 14
    for row in [head] + rows:
        for x, cell in zip(COLS, row):
            page.insert_text((x, y), cell, fontsize=11)
        y += 18
    y += 14
    for line in footer:
        page.insert_text((50, y), line, fontsize=11)
        y += 18
    doc.save(path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    vendor = "Vendor: Nilgiri Industrial Supplies"
    full_head = ("Item Description", "Qty", "Unit Price", "Amount")
    make_pdf(OUT / "PO-5001.pdf", "PURCHASE ORDER",
             ["PO Number: PO-5001", vendor, "Order Date: 2026-10-10", "Delivery Date: 2026-10-30", "Currency: INR"],
             full_head,
             [("Brass Valve 1in", "20", "450.00", "9,000.00"), ("Pressure Gauge", "10", "1,250.00", "12,500.00"),
              ("Teflon Tape Roll", "100", "35.00", "3,500.00")],
             ["Subtotal: 25,000.00", "GST: 4,500.00", "Grand Total: 29,500.00"])
    make_pdf(OUT / "GRN-6001.pdf", "GOODS RECEIPT NOTE",
             ["GRN No: GRN-6001", "PO Number: PO-5001", vendor, "Receipt Date: 2026-10-20"],
             ("Item Description", "Qty Received"),
             [("Brass Valve 1in", "20"), ("Pressure Gauge", "8"), ("Teflon Tape Roll", "100")],
             ["Received by: Stores"])
    make_pdf(OUT / "INV-7001.pdf", "TAX INVOICE",
             ["Invoice No: INV-7001", "PO Number: PO-5001", vendor, "Invoice Date: 2026-10-21", "Currency: INR"],
             full_head,
             [("Brass Valve 1in", "20", "450.00", "9,000.00"), ("Pressure Gauge", "8", "1,250.00", "10,000.00"),
              ("Teflon Tape Roll", "100", "35.00", "3,500.00")],
             ["Subtotal: 22,500.00", "GST: 4,050.00", "Grand Total: 26,550.00"])
    make_pdf(OUT / "INV-7002.pdf", "TAX INVOICE",
             ["Invoice No: INV-7002", "PO Number: PO-5001", vendor, "Invoice Date: 2026-10-22", "Currency: INR"],
             full_head,
             [("Brass Valve 1in", "20", "495.00", "9,900.00"), ("Pressure Gauge", "10", "1,250.00", "12,500.00"),
              ("Teflon Tape Roll", "100", "35.00", "3,500.00")],
             ["Subtotal: 25,900.00", "GST: 4,662.00", "Grand Total: 30,562.00"])
    print("Created PDFs in", OUT)


if __name__ == "__main__":
    main()
