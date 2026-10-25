from datetime import date

import pytest

from src import db, demo_data
from src.alerts import build_alerts
from src.analysis import assess_open_orders, match_all_invoices
from src.risk_model import train_models

TODAY = date(2026, 10, 22)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("d") / "demo.db")
    with db.get_conn(path) as c:
        demo_data.load_demo(c, TODAY)
    return path


def test_demo_counts(loaded):
    with db.get_conn(loaded) as c:
        counts = db.counts(c)
    assert counts == {"Purchase orders": 14, "Receipts (GRN)": 6, "Invoices": 8,
                      "Historical orders": 1200, "Inventory items": 8}


def test_each_deliberate_error_is_found(loaded):
    with db.get_conn(loaded) as c:
        results = {m["invoice_id"]: m for m in match_all_invoices(c)}
    codes = lambda i: {d["code"] for d in results[i]["result"]["discrepancies"]}
    assert results[1]["result"]["status"] == "Matched"                                   # INV-4001 clean
    assert "PRICE_HIGHER_THAN_PO" in codes(2)                                             # INV-4002
    assert "QTY_EXCEEDS_RECEIPT" in codes(3)                                              # INV-4003
    assert "GRN_NOT_FOUND" in codes(4)                                                    # INV-4004
    assert results[5]["result"]["status"] == "Matched" and "DUPLICATE_INVOICE" in codes(6)  # INV-4005 twice
    assert {"ITEM_NOT_ON_PO", "INVOICE_LINE_ARITHMETIC"} <= codes(7)                      # INV-4006
    assert "INVOICE_SUBTOTAL_MISMATCH" in codes(8)                                        # INV-4007


def test_open_lines_and_assessment(loaded):
    with db.get_conn(loaded) as c:
        bundle = train_models(db.load_history(c))
        open_lines = db.get_open_po_lines(c)
        a = assess_open_orders(c, bundle, TODAY)
    assert set(open_lines["po_number"]) == {"PO-2003", "PO-2004", "PO-2008", "PO-2009", "PO-2010", "PO-2011",
                                            "PO-2012", "PO-2013", "PO-2014"}
    assert len(a["risk"]) == len(open_lines) and a["risk"]["p_risk"].between(0, 1).all()
    assert a["risk"]["p_risk"].is_monotonic_decreasing
    sh = a["shortage"].set_index("material")
    assert sh.loc["Lubricant Oil 5L", "risk"] == "Not assessed"       # demand missing on purpose
    assert sh.loc["Packaging Boxes", "risk"] == "Low"
    assert sh.loc["Hex Nuts M8", "risk"] == "High"                    # PO-2013 arrives after stock runs out


def test_alerts_are_ranked_and_complete(loaded):
    with db.get_conn(loaded) as c:
        bundle = train_models(db.load_history(c))
        a = assess_open_orders(c, bundle, TODAY)
        alerts = build_alerts(match_all_invoices(c), a["risk"], a["reasons"], a["shortage"], TODAY)
    assert alerts["score"].is_monotonic_decreasing
    assert {"Invoice", "Delivery", "Shortage"} <= set(alerts["category"])
    assert alerts["action"].str.len().min() > 10
    overdue = alerts[alerts["title"].str.contains("overdue")]
    assert any("PO-2003" in t for t in overdue["title"])               # partial, past its promised date
