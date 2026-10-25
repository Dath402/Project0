"""Material-shortage risk from open orders, stock and demand.

Plain day-by-day stock projection, three scenarios:
  planned   every open order arrives on its promised date, in full
  expected  late probability x typical delay, incomplete probability x typical shortfall
  stress    every open order arrives late AND short by the supplier's typical amounts
Stock on day t = stock + deliveries arrived by day t - daily demand x t.
A stock-out is the first day this falls below the safety stock.
"""
import math
from datetime import date

import pandas as pd

HORIZON_DAYS = 30
RESULT_COLUMNS = ["material", "stock", "daily_demand", "days_of_cover", "incoming_qty", "stockout_day_planned",
                  "stockout_day_expected", "stockout_day_stress", "shortage_units_expected", "risk", "reason"]


def _isnum(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _arrivals(lines: pd.DataFrame, as_of: date, scenario: str, stats: dict,
              default_delay: float, default_short: float) -> list[tuple[int, float]]:
    out = []
    for r in lines.itertuples():
        promised = pd.to_datetime(r.promised_date).date() if _isnum(r.promised_date) and pd.notna(r.promised_date) else as_of
        day = max((promised - as_of).days, 0)
        qty = float(r.remaining_qty)
        s = stats.get(r.supplier, {})
        delay = s.get("avg_delay_days", default_delay)
        short = s.get("avg_shortfall", default_short)
        if scenario == "expected":
            day += round(float(r.p_late) * delay)
            qty *= 1 - float(r.p_incomplete) * short
        elif scenario == "stress":
            day += math.ceil(delay)
            qty *= 1 - short
        out.append((day, qty))
    return out


def project_stock(stock: float, demand: float, arrivals: list[tuple[int, float]],
                  horizon: int = HORIZON_DAYS) -> list[float]:
    """Stock level at the end of day 0..horizon (arrivals count from the start of their day)."""
    levels = []
    for t in range(horizon + 1):
        arrived = sum(q for d, q in arrivals if d <= t)
        levels.append(stock + arrived - demand * t)
    return levels


def scenario_levels(lines: pd.DataFrame, stock: float, demand: float, stats: dict, as_of: date,
                    default_delay: float = 3.0, default_short: float = 0.2,
                    horizon: int = HORIZON_DAYS) -> dict[str, list[float]]:
    """Projected stock for the planned / expected / stress scenarios (used by the charts)."""
    return {s: project_stock(stock, demand,
                             _arrivals(lines, as_of, s, stats, default_delay, default_short) if len(lines) else [],
                             horizon)
            for s in ("planned", "expected", "stress")}


def _first_below(levels: list[float], safety: float) -> int | None:
    for t, level in enumerate(levels):
        if t > 0 and level < safety:
            return t
    return None


def assess_shortage(open_lines: pd.DataFrame, inventory: pd.DataFrame, stats: dict, as_of: date,
                    default_delay: float = 3.0, default_short: float = 0.2,
                    horizon: int = HORIZON_DAYS, safety_days: float = 0.0) -> pd.DataFrame:
    """open_lines needs: supplier, material, remaining_qty, promised_date, p_late, p_incomplete."""
    rows = []
    for inv in inventory.itertuples():
        lines = open_lines[open_lines["material"] == inv.material] if len(open_lines) else open_lines
        incoming = float(lines["remaining_qty"].sum()) if len(lines) else 0.0
        row = {"material": inv.material, "stock": inv.stock, "daily_demand": inv.daily_demand,
               "days_of_cover": None, "incoming_qty": incoming, "stockout_day_planned": None,
               "stockout_day_expected": None, "stockout_day_stress": None,
               "shortage_units_expected": None, "risk": "Not assessed", "reason": ""}
        if not _isnum(inv.stock) or not _isnum(inv.daily_demand) or inv.daily_demand <= 0:
            row["reason"] = "Stock or daily demand is missing, so a shortage cannot be estimated."
            rows.append(row)
            continue
        safety = safety_days * inv.daily_demand
        row["days_of_cover"] = round(inv.stock / inv.daily_demand, 1)
        levels = scenario_levels(lines, inv.stock, inv.daily_demand, stats, as_of,
                                 default_delay, default_short, horizon)
        for s in levels:
            row[f"stockout_day_{s}"] = _first_below(levels[s], safety)
        row["shortage_units_expected"] = round(max(0.0, -min(levels["expected"])), 1)
        if row["stockout_day_planned"] is not None:
            row["risk"] = "High"
            row["reason"] = (f"Stock runs out on day {row['stockout_day_planned']} even if every open order "
                             f"arrives on time. Open orders do not cover demand.")
        elif row["stockout_day_expected"] is not None:
            row["risk"] = "High"
            row["reason"] = (f"Stock runs out on day {row['stockout_day_expected']} when expected "
                             f"delays and shortfalls are included.")
        elif row["stockout_day_stress"] is not None:
            row["risk"] = "Medium"
            row["reason"] = (f"Fine on expectation, but stock runs out on day {row['stockout_day_stress']} "
                             f"if all open orders are late and short.")
        else:
            row["risk"] = "Low"
            row["reason"] = f"Stock lasts the next {horizon} days even in the stress case."
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
