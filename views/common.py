"""Shared helpers for the dashboard pages."""
import html
from datetime import date

import pandas as pd
import streamlit as st

from src import db
from src.alerts import build_alerts
from src.analysis import assess_open_orders, match_all_invoices
from src.risk_model import train_models

CSS = """
<style>
:root { --accent: #6366f1; --accent-2: #06b6d4; }
.block-container { padding-top: 2rem; max-width: 1100px; }
.hero h1 {
    font-size: 2rem; font-weight: 800; margin: 0 0 .2rem 0;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { opacity: .75; margin: 0 0 1rem 0; }
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
    padding: .7rem 1rem; background: rgba(99,102,241,.05);
}
.field-card {
    border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
    padding: .6rem 1rem; margin-bottom: .6rem; border-left: 4px solid var(--accent);
}
.field-card .lbl { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6; }
.field-card .val { font-size: 1.05rem; font-weight: 600; word-break: break-word; }
.totals { border-radius: 12px; padding: .8rem 1rem; background: rgba(6,182,212,.07);
          border: 1px solid rgba(6,182,212,.3); margin-top: .8rem; }
.totals div { display: flex; justify-content: space-between; padding: .15rem 0; }
.totals .grand { border-top: 1px solid rgba(128,128,128,.3); margin-top: .35rem;
                 padding-top: .45rem; font-weight: 800; font-size: 1.1rem; }
.status { border-radius: 12px; padding: .8rem 1rem; font-weight: 700; margin: .4rem 0 .8rem 0; }
.status.ok { background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.45); }
.status.warn { background: rgba(234,179,8,.12); border: 1px solid rgba(234,179,8,.5); }
.status.bad { background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.5); }
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, var(--accent), var(--accent-2)); border: 0;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def esc(value) -> str:
    """Escape untrusted text before it goes into HTML."""
    return html.escape("" if value is None else str(value))


def field_card(label: str, value) -> str:
    shown = esc(value) if value not in (None, "") else "—"
    return f'<div class="field-card"><div class="lbl">{esc(label)}</div><div class="val">{shown}</div></div>'


def money(currency, value) -> str:
    return esc(f"{currency or ''} {value}".strip()) if value is not None else "—"


def status_box(status: str) -> None:
    css = "ok" if status == "Matched" else "warn" if status.startswith("Matched with") else "bad"
    st.markdown(f'<div class="status {css}">{esc(status)}</div>', unsafe_allow_html=True)


def _history_fingerprint(conn) -> str:
    row = conn.execute("SELECT COUNT(*), COALESCE(SUM(actual_lead_days),0), COALESCE(SUM(quantity_received),0),"
                       " COALESCE(SUM(promised_lead_days),0) FROM history_orders").fetchone()
    return "|".join(str(x) for x in row)


@st.cache_resource(show_spinner="Training prediction models on the order history...")
def _train(fingerprint: str):          # fingerprint only keys the cache; data is read from the database
    with db.get_conn() as conn:
        return train_models(db.load_history(conn))


def get_context() -> dict:
    """Everything the dashboard pages need, computed from the current database."""
    as_of = date.today()
    ctx = {"as_of": as_of, "bundle": None, "bundle_error": None, "analysis": None,
           "matches": [], "alerts": pd.DataFrame(columns=["score", "category", "severity", "title", "action", "ref"])}
    with db.get_conn() as conn:
        ctx["counts"] = db.counts(conn)
        ctx["is_demo"] = bool(db.get_meta(conn, "demo_data", False))
        ctx["matches"] = match_all_invoices(conn)
        fp = _history_fingerprint(conn)
        if ctx["counts"]["Historical orders"] > 0:
            try:
                ctx["bundle"] = _train(fp)
            except ValueError as exc:
                ctx["bundle_error"] = str(exc)
        else:
            ctx["bundle_error"] = "No historical orders loaded yet. Open the Data page."
        if ctx["bundle"] is not None:
            ctx["analysis"] = assess_open_orders(conn, ctx["bundle"], as_of)
    if ctx["analysis"] is not None:
        a = ctx["analysis"]
        ctx["alerts"] = build_alerts(ctx["matches"], a["risk"], a["reasons"], a["shortage"], as_of)
    else:
        ctx["alerts"] = build_alerts(ctx["matches"], pd.DataFrame(columns=[
            "po_number", "supplier", "material", "promised_date", "p_late", "p_incomplete", "p_risk", "risk_level"]),
            {}, pd.DataFrame(columns=["material", "risk", "reason"]), as_of)
    return ctx
