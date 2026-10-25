import pandas as pd
import streamlit as st

from src import db, demo_data
from views.common import get_context, header


def render() -> None:
    header("Procurement dashboard", "Document verification and supply risk in one place.")
    ctx = get_context()
    counts = ctx["counts"]
    if counts["Purchase orders"] == 0 and counts["Historical orders"] == 0:
        st.info("The database is empty. Load the synthetic demo data to explore every feature, "
                "or add your own on the Data page.")
        if st.button("Load demo data", type="primary"):
            with db.get_conn() as conn:
                demo_data.load_demo(conn)
            st.rerun()
        return
    if ctx["is_demo"]:
        st.caption("Showing SYNTHETIC demo data. It is not real company data.")

    matches = ctx["matches"]
    on_hold = sum(1 for m in matches if m["result"]["status"].startswith("Hold"))
    analysis = ctx["analysis"]
    high_orders = int((analysis["risk"]["risk_level"] == "High").sum()) if analysis and len(analysis["risk"]) else 0
    short = int(analysis["shortage"]["risk"].isin(["High", "Medium"]).sum()) if analysis else 0
    a, b, c, d = st.columns(4)
    a.metric("Invoices on hold", f"{on_hold} / {len(matches)}")
    b.metric("High-risk orders", high_orders)
    c.metric("Materials at shortage risk", short)
    d.metric("Open alerts", len(ctx["alerts"]))

    st.subheader("Top alerts")
    top = ctx["alerts"].head(5)
    if top.empty:
        st.success("No alerts right now.")
    for r in top.itertuples():
        with st.container(border=True):
            st.markdown(f"**{r.severity} · {r.category}**")
            st.write(r.title)
            st.caption(f"Suggested follow-up: {r.action}")

    left, right = st.columns(2)
    with left:
        st.subheader("Invoice status")
        if matches:
            s = pd.Series([m["result"]["status"] for m in matches]).value_counts()
            st.bar_chart(s)
    with right:
        st.subheader("Delivery risk")
        if analysis and len(analysis["risk"]):
            st.bar_chart(analysis["risk"]["risk_level"].value_counts())
        elif ctx["bundle_error"]:
            st.info(ctx["bundle_error"])
