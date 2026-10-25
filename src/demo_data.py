"""SYNTHETIC demo data (not real company data).

Gives you a complete scenario to test every feature: 12 months of order history,
open purchase orders, receipts, invoices with deliberately introduced
discrepancies, and stock/demand figures.
"""
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from . import db
from .extract_fields import empty_document

# supplier: (late rate, incomplete rate, materials it supplies)
SUPPLIERS = {
    "Kerala Steel Works": (0.10, 0.05, ["Steel Bolts M8", "Hex Nuts M8", "Cutting Blades"]),
    "Malabar Fasteners": (0.20, 0.10, ["Steel Bolts M8", "Hex Nuts M8", "Packaging Boxes"]),
    "Coastal Electricals": (0.35, 0.15, ["Electric Switch", "Copper Wire 2mm", "Packaging Boxes"]),
    "Southern Polymers": (0.50, 0.25, ["PVC Pipe 1in", "Lubricant Oil 5L", "Packaging Boxes"]),
    "Western Ghats Tools": (0.25, 0.30, ["Cutting Blades", "Lubricant Oil 5L", "Electric Switch"]),
    "Eastern Packaging": (0.15, 0.08, ["Packaging Boxes", "PVC Pipe 1in", "Steel Bolts M8"]),
}
MATERIALS = {  # typical order quantity, unit price (INR)
    "Steel Bolts M8": (500, 2.5), "Hex Nuts M8": (600, 1.8), "Copper Wire 2mm": (200, 85.0),
    "PVC Pipe 1in": (150, 40.0), "Packaging Boxes": (400, 12.0), "Electric Switch": (250, 35.0),
    "Cutting Blades": (120, 220.0), "Lubricant Oil 5L": (80, 640.0),
}
TAX_RATE = Decimal("0.18")


def generate_history(today: date, n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Past orders with outcomes. Lateness depends on supplier, lead time, season and order size."""
    rng = np.random.default_rng(seed)
    names = list(SUPPLIERS)
    rows = []
    for i in range(n):
        supplier = names[rng.integers(len(names))]
        base_late, base_inc, mats = SUPPLIERS[supplier]
        material = mats[rng.integers(len(mats))]
        typical_qty, price = MATERIALS[material]
        qty = max(10, int(typical_qty * rng.lognormal(0, 0.45)))
        ratio = qty / typical_qty
        order_date = today - timedelta(days=int(rng.integers(60, 760)))
        promised = int(rng.integers(5, 35))
        busy = order_date.month in (11, 12)
        p_late = float(np.clip(base_late + 0.15 * (promised < 10) - 0.05 * (promised > 25)
                               + 0.12 * busy + 0.05 * (ratio > 1.5), 0.02, 0.9))
        p_inc = float(np.clip(base_inc + 0.10 * (ratio > 1.5) + 0.05 * busy, 0.01, 0.8))
        late = rng.random() < p_late
        actual = promised + 1 + int(rng.poisson(3)) if late else max(1, promised - int(rng.integers(0, 3)))
        received = qty * (1 - rng.uniform(0.05, 0.35)) if rng.random() < p_inc else qty
        rows.append({"po_number": f"H-{i + 1:04d}", "supplier": supplier, "material": material,
                     "order_date": order_date.isoformat(), "quantity_ordered": qty,
                     "unit_price": price, "promised_lead_days": promised,
                     "actual_lead_days": actual, "quantity_received": round(received, 1)})
    return pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)


def _money(x: Decimal) -> str:
    return str(x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _po(number, supplier, today, order_off, delivery_off, items) -> dict:
    d = empty_document("Purchase Order")
    lines = [{"name": n, "quantity": str(q), "unit_price": _money(Decimal(str(p))),
              "line_total": _money(Decimal(q) * Decimal(str(p)))} for n, q, p in items]
    sub = sum((Decimal(l["line_total"]) for l in lines), Decimal(0))
    tax = (sub * TAX_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    d.update(document_id=number, vendor=supplier, currency="INR", items=lines,
             document_date=(today + timedelta(days=order_off)).isoformat(),
             delivery_date=(today + timedelta(days=delivery_off)).isoformat(),
             subtotal=_money(sub), tax=_money(tax), total=_money(sub + tax), source_filename=f"{number}.pdf")
    return d


def _grn(number, po_number, supplier, today, off, items) -> dict:
    d = empty_document("Delivery Receipt / GRN")
    d.update(document_id=number, po_reference=po_number, vendor=supplier,
             document_date=(today + timedelta(days=off)).isoformat(), source_filename=f"{number}.pdf",
             items=[{"name": n, "quantity": str(q), "unit_price": None, "line_total": None} for n, q in items])
    return d


def _invoice(number, po_number, supplier, today, off, items, subtotal=None, tax=None, total=None) -> dict:
    """items: (name, qty, price) or (name, qty, price, line_total_override)."""
    d = empty_document("Invoice")
    lines = []
    for it in items:
        n, q, p = it[:3]
        line = it[3] if len(it) > 3 else _money(Decimal(q) * Decimal(str(p)))
        lines.append({"name": n, "quantity": str(q), "unit_price": _money(Decimal(str(p))), "line_total": line})
    sub = sum((Decimal(l["line_total"]) for l in lines), Decimal(0))
    sub = Decimal(subtotal) if subtotal is not None else sub
    t = Decimal(tax) if tax is not None else (sub * TAX_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    tot = Decimal(total) if total is not None else sub + t
    d.update(document_id=number, po_reference=po_number, vendor=supplier, currency="INR", items=lines,
             document_date=(today + timedelta(days=off)).isoformat(), subtotal=_money(sub), tax=_money(t),
             total=_money(tot), source_filename=f"{number}.pdf")
    return d


def build_documents(today: date) -> dict:
    K, M, C, S, W, E = ("Kerala Steel Works", "Malabar Fasteners", "Coastal Electricals",
                        "Southern Polymers", "Western Ghats Tools", "Eastern Packaging")
    pos = [
        _po("PO-2001", K, today, -30, -12, [("Steel Bolts M8", 400, 2.5), ("Hex Nuts M8", 400, 1.8)]),
        _po("PO-2002", M, today, -28, -10, [("Steel Bolts M8", 300, 2.6), ("Hex Nuts M8", 300, 1.85)]),
        _po("PO-2003", C, today, -26, -4, [("Electric Switch", 200, 35), ("Copper Wire 2mm", 100, 85)]),
        _po("PO-2004", S, today, -12, 9, [("PVC Pipe 1in", 150, 40)]),
        _po("PO-2005", E, today, -20, -6, [("Packaging Boxes", 300, 12)]),
        _po("PO-2006", W, today, -22, -5, [("Cutting Blades", 100, 220), ("Lubricant Oil 5L", 40, 640)]),
        _po("PO-2007", K, today, -24, -8, [("Steel Bolts M8", 250, 2.5)]),
        _po("PO-2008", S, today, -8, 6, [("PVC Pipe 1in", 400, 40)]),
        _po("PO-2009", C, today, -6, 5, [("Copper Wire 2mm", 300, 85)]),
        _po("PO-2010", K, today, -5, 18, [("Steel Bolts M8", 600, 2.5)]),
        _po("PO-2011", W, today, -7, 12, [("Cutting Blades", 150, 220)]),
        _po("PO-2012", E, today, -4, 9, [("Packaging Boxes", 500, 12)]),
        _po("PO-2013", M, today, -2, 20, [("Hex Nuts M8", 800, 1.8), ("Steel Bolts M8", 400, 2.6)]),
        _po("PO-2014", W, today, -6, 8, [("Lubricant Oil 5L", 100, 640)]),
    ]
    grns = [
        _grn("GRN-3001", "PO-2001", K, today, -11, [("Steel Bolts M8", 400), ("Hex Nuts M8", 400)]),
        _grn("GRN-3002", "PO-2002", M, today, -9, [("Steel Bolts M8", 300), ("Hex Nuts M8", 300)]),
        _grn("GRN-3003", "PO-2003", C, today, -3, [("Electric Switch", 160), ("Copper Wire 2mm", 100)]),
        _grn("GRN-3005", "PO-2005", E, today, -5, [("Packaging Boxes", 300)]),
        _grn("GRN-3006", "PO-2006", W, today, -4, [("Cutting Blades", 100), ("Lubricant Oil 5L", 40)]),
        _grn("GRN-3007", "PO-2007", K, today, -7, [("Steel Bolts M8", 250)]),
    ]
    invoices = [
        # 4001 clean
        _invoice("INV-4001", "PO-2001", K, today, -9, [("Steel Bolts M8", 400, 2.5), ("Hex Nuts M8", 400, 1.8)]),
        # 4002 higher unit price than PO
        _invoice("INV-4002", "PO-2002", M, today, -8, [("Steel Bolts M8", 300, 2.95), ("Hex Nuts M8", 300, 1.85)]),
        # 4003 bills 200 switches but only 160 arrived
        _invoice("INV-4003", "PO-2003", C, today, -2, [("Electric Switch", 200, 35), ("Copper Wire 2mm", 100, 85)]),
        # 4004 invoice arrived before any goods were received
        _invoice("INV-4004", "PO-2004", S, today, -5, [("PVC Pipe 1in", 150, 40)]),
        # 4005 correct, then submitted again
        _invoice("INV-4005", "PO-2005", E, today, -4, [("Packaging Boxes", 300, 12)]),
        _invoice("INV-4005", "PO-2005", E, today, -3, [("Packaging Boxes", 300, 12)]),
        # 4006 extra item not on the PO and one wrong line total
        _invoice("INV-4006", "PO-2006", W, today, -3,
                 [("Cutting Blades", 100, 220), ("Lubricant Oil 5L", 40, 640, "25000.00"),
                  ("Safety Gloves", 20, 150)]),
        # 4007 subtotal does not match the lines
        _invoice("INV-4007", "PO-2007", K, today, -6, [("Steel Bolts M8", 250, 2.5)], subtotal="650.00"),
    ]
    return {"pos": pos, "grns": grns, "invoices": invoices}


def build_inventory() -> pd.DataFrame:
    return pd.DataFrame([
        ("Steel Bolts M8", 900, 40), ("Hex Nuts M8", 700, 45), ("Copper Wire 2mm", 120, 18),
        ("PVC Pipe 1in", 90, 12), ("Packaging Boxes", 1500, 40), ("Electric Switch", 60, 10),
        ("Cutting Blades", 70, 6), ("Lubricant Oil 5L", 45, None),      # demand unknown on purpose
    ], columns=db.INVENTORY_COLUMNS)


def load_demo(conn, today: date | None = None, seed: int = 7) -> dict:
    """Replace everything in the database with the synthetic demo scenario."""
    today = today or date.today()
    db.clear_all(conn)
    db.save_history(conn, generate_history(today, seed=seed))
    db.save_inventory(conn, build_inventory())
    docs = build_documents(today)
    for doc in docs["pos"] + docs["grns"] + docs["invoices"]:
        db.save_document(conn, doc)
    db.set_meta(conn, "demo_data", True)
    db.set_meta(conn, "loaded_on", today.isoformat())
    return db.counts(conn)
