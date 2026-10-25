import pandas as pd
import streamlit as st

from src import db, demo_data
from views.common import header

HISTORY_TEMPLATE = (",".join(db.HISTORY_COLUMNS) + "\n"
                    "H-0001,ABC Supplies,Steel Bolts M8,2025-01-15,500,2.5,14,16,500\n")
INVENTORY_TEMPLATE = ",".join(db.INVENTORY_COLUMNS) + "\nSteel Bolts M8,900,40\n"


def render() -> None:
    header("Data", "Load demo data or your own history and stock figures.")
    with db.get_conn() as conn:
        counts = db.counts(conn)
    st.dataframe(pd.DataFrame([counts]), width="stretch", hide_index=True)
    st.caption("On free hosting the database file can reset when the app restarts.")

    st.subheader("Demo data")
    st.write("Replaces everything with a synthetic scenario: 1,200 past orders, 14 purchase orders, receipts, "
             "8 invoices with deliberate errors, and stock figures.")
    if st.checkbox("I understand this replaces all current data") and st.button("Load demo data", type="primary"):
        with db.get_conn() as conn:
            demo_data.load_demo(conn)
        st.success("Demo data loaded.")
        st.rerun()

    st.subheader("Your own order history (CSV)")
    st.download_button("Download history template", HISTORY_TEMPLATE, "history_template.csv", "text/csv")
    st.caption("Needs at least 60 completed orders. Actual lead time and received quantity are outcomes of past orders.")
    up = st.file_uploader("Upload history CSV", type=["csv"], key="hist")
    if up is not None and st.button("Replace history"):
        try:
            with db.get_conn() as conn:
                n = db.save_history(conn, pd.read_csv(up))
            st.success(f"Loaded {n} historical orders.")
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")

    st.subheader("Stock and demand (CSV, optional)")
    st.download_button("Download inventory template", INVENTORY_TEMPLATE, "inventory_template.csv", "text/csv")
    up2 = st.file_uploader("Upload inventory CSV", type=["csv"], key="inv")
    if up2 is not None and st.button("Replace inventory"):
        try:
            with db.get_conn() as conn:
                n = db.save_inventory(conn, pd.read_csv(up2))
            st.success(f"Loaded {n} inventory items.")
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")

    st.subheader("Reset")
    if st.checkbox("Delete everything") and st.button("Reset database"):
        with db.get_conn() as conn:
            db.clear_all(conn)
        st.success("Database cleared.")
        st.rerun()
