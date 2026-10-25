"""Glue that runs matching, prediction and shortage assessment over the database."""
from datetime import date

import pandas as pd

from . import db
from .matching import Tolerances, three_way_match
from .risk_model import (explain_order, predict_orders, supplier_outcome_stats)
from .shortage import assess_shortage


def match_all_invoices(conn, tol: Tolerances = Tolerances()) -> list[dict]:
    """Run the three-way match for every saved invoice."""
    headers = db.all_invoice_headers(conn)
    out = []
    for inv in db.list_invoices(conn):
        ref = inv.get("po_reference")
        po = db.get_po(conn, ref) if ref else None
        grns = db.get_grns_for_po(conn, ref) if ref else []
        result = three_way_match(po, grns, inv, headers, tol)
        out.append({"invoice_id": inv["id"], "invoice": inv["document_id"], "po": ref or "(none)",
                    "supplier": inv["vendor"], "result": result})
    return out


def _targets_from_open_lines(open_lines: pd.DataFrame, history: pd.DataFrame, as_of: date) -> pd.DataFrame:
    order = pd.to_datetime(open_lines["order_date"], errors="coerce").fillna(pd.Timestamp(as_of))
    promised = pd.to_datetime(open_lines["promised_date"], errors="coerce")
    lead = (promised - order).dt.days
    typical_lead = float(history["promised_lead_days"].median()) if len(history) else 14.0
    return pd.DataFrame({"supplier": open_lines["supplier"], "material": open_lines["material"],
                         "order_date": order, "quantity": open_lines["quantity_ordered"],
                         "unit_price": open_lines["unit_price"],
                         "promised_lead_days": lead.fillna(typical_lead).clip(lower=1)})


def assess_open_orders(conn, bundle: dict, as_of: date | None = None) -> dict:
    """Delivery-risk and shortage tables for every open PO line."""
    as_of = as_of or date.today()
    history = db.load_history(conn)
    open_lines = db.get_open_po_lines(conn)
    inventory = db.load_inventory(conn)
    empty = {"risk": pd.DataFrame(), "features": pd.DataFrame(), "reasons": {}, "explanations": {},
             "shortage": pd.DataFrame(), "as_of": as_of}
    if open_lines.empty:
        empty["shortage"] = assess_shortage(open_lines.assign(p_late=[], p_incomplete=[]), inventory, {}, as_of)
        return empty
    targets = _targets_from_open_lines(open_lines, history, as_of)
    scored, feats = predict_orders(bundle, targets, history)
    risk = open_lines.reset_index(drop=True).copy()
    risk["row"] = risk.index                      # position in the feature table
    for col in ("p_late", "p_incomplete", "p_risk", "risk_level"):
        risk[col] = scored[col]
    explanations, reasons = {}, {}
    for i, r in risk.iterrows():
        target = "late" if r["p_late"] >= r["p_incomplete"] else "incomplete"
        items = explain_order(bundle, feats.iloc[i], target)
        explanations[(r["po_number"], r["material"])] = {"target": target, "items": items}
        reasons[(r["po_number"], r["material"])] = "; ".join(x["text"] for x in items[:2])
    stats, d_delay, d_short = supplier_outcome_stats(history)
    shortage = assess_shortage(risk, inventory, stats, as_of, d_delay, d_short)
    return {"risk": risk.sort_values("p_risk", ascending=False).reset_index(drop=True), "features": feats,
            "reasons": reasons, "explanations": explanations, "shortage": shortage, "as_of": as_of,
            "stats": stats, "defaults": (d_delay, d_short)}
