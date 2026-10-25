from datetime import date

import pandas as pd

from src.shortage import assess_shortage, project_stock

AS_OF = date(2026, 10, 22)


def lines(rows):
    return pd.DataFrame(rows, columns=["supplier", "material", "remaining_qty", "promised_date", "p_late", "p_incomplete"])


def inv(stock, demand, material="Bolt"):
    return pd.DataFrame([{"material": material, "stock": stock, "daily_demand": demand}])


def test_project_stock_arithmetic():
    levels = project_stock(100, 10, [(5, 50)], horizon=10)
    assert levels[0] == 100 and levels[4] == 60 and levels[5] == 100 and levels[10] == 50


def test_low_risk_with_plenty_of_stock():
    r = assess_shortage(lines([]), inv(10_000, 10), {}, AS_OF)
    assert r.loc[0, "risk"] == "Low" and r.loc[0, "days_of_cover"] == 1000


def test_high_risk_when_orders_do_not_cover_demand():
    r = assess_shortage(lines([]), inv(50, 10), {}, AS_OF)
    assert r.loc[0, "risk"] == "High" and r.loc[0, "stockout_day_planned"] == 6
    assert "even if every open order" in r.loc[0, "reason"]


def test_delay_turns_a_safe_plan_into_high_risk():
    open_lines = lines([("S", "Bolt", 500, date(2026, 10, 30), 0.9, 0.0)])
    r = assess_shortage(open_lines, inv(100, 10), {"S": {"avg_delay_days": 6, "avg_shortfall": 0.2}}, AS_OF)
    # stock alone lasts until day 10; on time (day 8) is fine, but 90% x 6 days late = day 13 is not
    assert pd.isna(r.loc[0, "stockout_day_planned"])
    assert r.loc[0, "stockout_day_expected"] == 11 and r.loc[0, "risk"] == "High"


def test_stress_only_case_is_medium():
    open_lines = lines([("S", "Bolt", 400, date(2026, 10, 30), 0.0, 0.0)])
    r = assess_shortage(open_lines, inv(100, 10), {"S": {"avg_delay_days": 5, "avg_shortfall": 0.0}}, AS_OF)
    assert r.loc[0, "risk"] == "Medium" and pd.isna(r.loc[0, "stockout_day_expected"])


def test_missing_demand_is_not_assessed_not_guessed():
    r = assess_shortage(lines([]), pd.DataFrame([{"material": "Bolt", "stock": 100, "daily_demand": None}]), {}, AS_OF)
    assert r.loc[0, "risk"] == "Not assessed" and "missing" in r.loc[0, "reason"]


def test_overdue_order_counts_from_today():
    open_lines = lines([("S", "Bolt", 100, date(2026, 10, 1), 0.0, 0.0)])
    r = assess_shortage(open_lines, inv(10, 10), {}, AS_OF)
    assert r.loc[0, "incoming_qty"] == 100
