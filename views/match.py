from decimal import Decimal

import pandas as pd
import streamlit as st

from src import db
from src.analysis import match_all_invoices
from src.matching import Tolerances, three_way_match
from views.common import header, status_box


def render() -> None:
    header("Three-way match", "Compare each invoice with its purchase order and goods receipts.")
    with st.expander("Tolerances"):
        price_pct = st.number_input("Allowed unit-price difference (%)", 0.0, 20.0, 1.0, 0.5)
        qty_abs = st.number_input("Allowed quantity difference (units)", 0.0, 1000.0, 0.0, 1.0)
    tol = Tolerances(price_pct=Decimal(str(price_pct)), qty_abs=Decimal(str(qty_abs)))

    with db.get_conn() as conn:
        results = match_all_invoices(conn, tol)
        if not results:
            st.info("No invoices saved yet. Extract and save documents, or load demo data on the Data page.")
            return
        st.subheader("All invoices")
        st.dataframe(pd.DataFrame([{"Record": m["invoice_id"], "Invoice": m["invoice"], "PO": m["po"],
                                    "Supplier": m["supplier"], "Status": m["result"]["status"],
                                    "Issues": len(m["result"]["discrepancies"])} for m in results]),
                     width="stretch", hide_index=True)

        labels = {m["invoice_id"]: f"#{m['invoice_id']} · {m['invoice']} → {m['po']}" for m in results}
        chosen = st.selectbox("Inspect one invoice", list(labels), format_func=labels.get)
        match = next(m for m in results if m["invoice_id"] == chosen)
        inv = db.get_invoice(conn, chosen)
        po = db.get_po(conn, inv["po_reference"]) if inv["po_reference"] else None
        grns = db.get_grns_for_po(conn, inv["po_reference"]) if inv["po_reference"] else []

    res = match["result"]
    status_box(res["status"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Invoice total", inv["total"] or "—")
    c2.metric("PO total", (po or {}).get("total") or "—")
    c3.metric("Supported by PO + receipts", res["supported_subtotal"] or "—")
    st.caption("“Supported” is the pre-tax value of quantities that are both ordered and received, at PO prices.")

    st.subheader("Discrepancies")
    if res["discrepancies"]:
        st.dataframe(pd.DataFrame(res["discrepancies"])[["severity", "message", "expected", "actual"]],
                     width="stretch", hide_index=True)
    else:
        st.success("No differences found.")
    for note in res["notes"]:
        st.info(note)

    st.subheader("Item comparison")
    if res["items"]:
        st.dataframe(pd.DataFrame(res["items"]).astype(object).where(lambda d: d.notna(), "—").astype(str),
                     width="stretch", hide_index=True)
    with st.expander("Linked documents"):
        st.write(f"Purchase order: {po['document_id'] if po else 'not found'}")
        st.write("Goods receipts: " + (", ".join(g["document_id"] for g in grns) if grns else "none recorded"))
