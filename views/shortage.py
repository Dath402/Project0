import pandas as pd
import streamlit as st

from src.shortage import HORIZON_DAYS, scenario_levels
from views.common import get_context, header


def render() -> None:
    header("Shortage risk", f"Will stock last the next {HORIZON_DAYS} days, given open orders and demand?")
    ctx = get_context()
    if ctx["analysis"] is None:
        st.info(ctx["bundle_error"] or "No analysis available.")
        return
    a = ctx["analysis"]
    sh = a["shortage"]
    if sh.empty:
        st.info("No inventory data yet. Add stock and demand on the Data page.")
        return
    order = {"High": 0, "Medium": 1, "Low": 2, "Not assessed": 3}
    sh = sh.sort_values("risk", key=lambda s: s.map(order), kind="stable")
    st.dataframe(sh[["material", "risk", "stock", "daily_demand", "days_of_cover", "incoming_qty",
                     "stockout_day_planned", "stockout_day_expected", "stockout_day_stress", "reason"]].rename(columns={
        "material": "Material", "risk": "Risk", "stock": "Stock", "daily_demand": "Demand/day",
        "days_of_cover": "Days of cover", "incoming_qty": "On order", "stockout_day_planned": "Out day (plan)",
        "stockout_day_expected": "Out day (expected)", "stockout_day_stress": "Out day (stress)",
        "reason": "Why"}), width="stretch", hide_index=True)
    st.caption("Planned: orders arrive on time and in full. Expected: delays and shortfalls weighted by the model. "
               "Stress: every open order is late and short by the supplier's usual amount.")

    assessed = sh[sh["risk"] != "Not assessed"]["material"].tolist()
    if not assessed:
        return
    material = st.selectbox("Projected stock for", assessed)
    inv = sh[sh["material"] == material].iloc[0]
    risk = a["risk"]
    lines = risk[risk["material"] == material] if len(risk) else risk
    d_delay, d_short = a.get("defaults", (3.0, 0.2))
    levels = scenario_levels(lines, inv["stock"], inv["daily_demand"], a.get("stats", {}), a["as_of"], d_delay, d_short)
    st.line_chart(pd.DataFrame(levels).rename_axis("Day"))
    st.caption("Below zero means demand cannot be met on that day.")
