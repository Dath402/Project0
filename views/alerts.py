import streamlit as st

from views.common import get_context, header


def render() -> None:
    header("Alerts and follow-up", "Everything that needs attention, most urgent first.")
    ctx = get_context()
    alerts = ctx["alerts"]
    if alerts.empty:
        st.success("No alerts. Load data or save documents to see results.")
        return
    c1, c2 = st.columns(2)
    cats = c1.multiselect("Type", sorted(alerts["category"].unique()), default=sorted(alerts["category"].unique()))
    sevs = c2.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    shown = alerts[alerts["category"].isin(cats) & alerts["severity"].isin(sevs)]
    st.caption(f"{len(shown)} of {len(alerts)} alerts")
    for r in shown.itertuples():
        with st.container(border=True):
            st.markdown(f"**{r.severity} · {r.category}**")
            st.write(r.title)
            st.caption(f"Suggested follow-up: {r.action}")
    st.download_button("Download alerts (CSV)", shown.drop(columns=["score"]).to_csv(index=False),
                       file_name="alerts.csv", mime="text/csv")
