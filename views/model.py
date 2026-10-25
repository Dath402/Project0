import pandas as pd
import streamlit as st

from src.risk_model import FEATURE_LABELS, MEDIUM_RISK
from views.common import get_context, header


def render() -> None:
    header("Model performance", "How well the late and incomplete predictors work on held-out history.")
    ctx = get_context()
    bundle = ctx["bundle"]
    if bundle is None:
        st.info(ctx["bundle_error"])
        return
    if ctx["is_demo"]:
        st.warning("The history is SYNTHETIC, so these scores describe the demo data generator, "
                   "not real supplier behaviour. Use real history to judge real performance.")
    st.write(f"Trained on the earliest {bundle['train_rows']} orders and tested on the latest "
             f"{bundle['test_rows']} (time-ordered split, no shuffling). An order is flagged when its "
             f"probability is at least {MEDIUM_RISK:.0%}.")
    for key, title in (("late", "Late delivery"), ("incomplete", "Incomplete delivery")):
        t = bundle["targets"][key]
        st.subheader(title)
        st.caption(f"Selected model: {t['chosen']} (best cross-validated ROC-AUC). "
                   f"{t['positive_rate']:.0%} of past orders had this problem.")
        df = pd.DataFrame(t["comparison"]).set_index("Model")
        st.dataframe(df.round(2), width="stretch")
        (tn, fp), (fn, tp) = t["confusion"]
        st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually fine", "Actually problem"],
                                  columns=["Predicted fine", "Predicted problem"]), width="stretch")
        if t["importance"]:
            imp = pd.DataFrame({"Importance": [v for _, v in t["importance"]]},
                               index=[FEATURE_LABELS[f] for f, _ in t["importance"]])
            st.bar_chart(imp, horizontal=True)
    st.info("Leakage rule: actual receipt dates and received quantities are used only as labels and to build "
            "supplier history from orders that finished before the order being scored.")
