import pandas as pd
import streamlit as st

from src.risk_model import explain_order
from views.common import get_context, header


def render() -> None:
    header("Delivery risk", "Open orders ranked by the chance they arrive late or incomplete.")
    ctx = get_context()
    if ctx["analysis"] is None:
        st.info(ctx["bundle_error"] or "No analysis available.")
        return
    risk = ctx["analysis"]["risk"]
    if risk.empty:
        st.success("There are no open order lines.")
        return
    if ctx["is_demo"]:
        st.caption("Predictions use SYNTHETIC history, so they only show how the system works.")

    table = risk[["po_number", "supplier", "material", "promised_date", "remaining_qty",
                  "p_late", "p_incomplete", "p_risk", "risk_level"]].copy()
    for col in ("p_late", "p_incomplete", "p_risk"):
        table[col] = (table[col] * 100).round(0)          # shown as whole percentages
    table = table.rename(columns={
        "po_number": "PO", "supplier": "Supplier", "material": "Material", "promised_date": "Promised",
        "remaining_qty": "Still due", "p_late": "Late", "p_incomplete": "Incomplete", "p_risk": "Overall",
        "risk_level": "Level"})
    pct = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={"Late": pct, "Incomplete": pct, "Overall": pct})
    st.caption("Overall = chance of being late or incomplete, assuming the two are independent. "
               "High ≥ 50%, Medium ≥ 30%.")

    labels = [f"{r.po_number} · {r.material}" for r in risk.itertuples()]
    pick = st.selectbox("Why is this order risky?", range(len(labels)), format_func=lambda i: labels[i])
    row = risk.iloc[pick]
    feats = ctx["analysis"]["features"].iloc[int(row["row"])]
    for target, title in (("late", "Late delivery"), ("incomplete", "Incomplete delivery")):
        st.markdown(f"**{title}: {row['p_late' if target == 'late' else 'p_incomplete']:.0%}**")
        items = explain_order(ctx["bundle"], feats, target, top=5)
        st.bar_chart(pd.DataFrame({"Effect on probability (points)": [x["effect"] * 100 for x in items]},
                                  index=[x["label"] for x in items]), horizontal=True)
        for x in items[:3]:
            st.caption("• " + x["text"])
