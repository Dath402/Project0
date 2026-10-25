"""Supplier delivery-risk prediction (late / incomplete) with scikit-learn.

Leakage rule: the actual receipt date and received quantity are used ONLY to
build training labels and to compute supplier history from orders that were
already completed BEFORE the order being scored. They are never input features
for the order itself.
"""
import math

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
INCOMPLETE_TOLERANCE = 0.02          # receiving less than 98% of the order counts as incomplete
MIN_HISTORY_ROWS = 60
SMOOTHING = 3                        # pseudo-orders that pull small samples toward the prior
PRIOR_LATE, PRIOR_INCOMPLETE, PRIOR_DELAY = 0.25, 0.15, 1.5
HIGH_RISK, MEDIUM_RISK = 0.50, 0.30
EPOCH = pd.Timestamp("1970-01-01")

FEATURES = ["log_quantity", "log_order_value", "promised_lead_days", "order_month", "order_weekday",
            "supplier_late_rate", "supplier_incomplete_rate", "supplier_avg_delay_days",
            "supplier_order_count", "material_late_rate", "qty_vs_material_median"]
FEATURE_LABELS = {
    "log_quantity": "Order quantity", "log_order_value": "Order value",
    "promised_lead_days": "Promised lead time (days)", "order_month": "Order month",
    "order_weekday": "Order weekday",
    "supplier_late_rate": "Supplier's past late rate",
    "supplier_incomplete_rate": "Supplier's past incomplete rate",
    "supplier_avg_delay_days": "Supplier's average past delay (days)",
    "supplier_order_count": "Supplier's completed past orders",
    "material_late_rate": "Past late rate for this material",
    "qty_vs_material_median": "Order size vs typical for this material",
}
CANDIDATES = {
    "Baseline": lambda: DummyClassifier(strategy="prior"),
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=10,
                                                    random_state=RANDOM_STATE, n_jobs=1),
}


def risk_level(probability: float) -> str:
    return "High" if probability >= HIGH_RISK else "Medium" if probability >= MEDIUM_RISK else "Low"


def _days(series) -> np.ndarray:
    return (pd.to_datetime(series) - EPOCH).dt.days.to_numpy()


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    """Add outcome columns (labels) to the raw history table."""
    h = history.copy()
    h["order_date"] = pd.to_datetime(h["order_date"])
    h["order_day"] = _days(h["order_date"])
    h["receipt_day"] = h["order_day"] + h["actual_lead_days"].astype(int)
    h["delay_days"] = (h["actual_lead_days"] - h["promised_lead_days"]).clip(lower=0)
    h["is_late"] = (h["actual_lead_days"] > h["promised_lead_days"]).astype(int)
    ordered = h["quantity_ordered"].replace(0, np.nan)
    h["shortfall_ratio"] = (1 - h["quantity_received"] / ordered).clip(lower=0).fillna(0)
    h["is_incomplete"] = (h["shortfall_ratio"] > INCOMPLETE_TOLERANCE).astype(int)
    return h


def _prior_stats(keys: np.ndarray, days: np.ndarray, hist: pd.DataFrame, key_col: str) -> dict:
    """For each row: totals over history rows with the same key that were RECEIVED before the order day."""
    n = np.zeros(len(keys))
    late, inc, delay = np.zeros(len(keys)), np.zeros(len(keys)), np.zeros(len(keys))
    for key in set(keys):
        g = hist[hist[key_col] == key].sort_values("receipt_day")
        if g.empty:
            continue
        rd = g["receipt_day"].to_numpy()
        cum = {c: np.concatenate([[0.0], np.cumsum(g[c].to_numpy(dtype=float))])
               for c in ("is_late", "is_incomplete", "delay_days")}
        mask = keys == key
        pos = np.searchsorted(rd, days[mask], side="left")      # strictly before the order day
        n[mask] = pos
        late[mask], inc[mask], delay[mask] = cum["is_late"][pos], cum["is_incomplete"][pos], cum["delay_days"][pos]
    return {"n": n, "late": late, "inc": inc, "delay": delay}


def compute_features(targets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Features for orders in `targets` (supplier, material, order_date, quantity, unit_price,
    promised_lead_days), using only history that was completed before each order date."""
    h = prepare_history(history)
    t = targets.reset_index(drop=True)
    days = _days(t["order_date"])
    sup = _prior_stats(t["supplier"].to_numpy(dtype=object), days, h, "supplier")
    mat = _prior_stats(t["material"].to_numpy(dtype=object), days, h, "material")
    k = SMOOTHING
    qty = t["quantity"].astype(float).to_numpy()
    price = t["unit_price"].astype(float).fillna(0).to_numpy()
    medians = h.groupby("material")["quantity_ordered"].median()
    typical = t["material"].map(medians).astype(float).fillna(pd.Series(qty)).replace(0, np.nan)
    dates = pd.to_datetime(t["order_date"])
    feats = pd.DataFrame({
        "log_quantity": np.log1p(qty),
        "log_order_value": np.log1p(qty * price),
        "promised_lead_days": t["promised_lead_days"].astype(float).to_numpy(),
        "order_month": dates.dt.month.to_numpy(),
        "order_weekday": dates.dt.weekday.to_numpy(),
        "supplier_late_rate": (sup["late"] + PRIOR_LATE * k) / (sup["n"] + k),
        "supplier_incomplete_rate": (sup["inc"] + PRIOR_INCOMPLETE * k) / (sup["n"] + k),
        "supplier_avg_delay_days": (sup["delay"] + PRIOR_DELAY * k) / (sup["n"] + k),
        "supplier_order_count": sup["n"],
        "material_late_rate": (mat["late"] + PRIOR_LATE * k) / (mat["n"] + k),
        "qty_vs_material_median": (qty / typical.to_numpy()),
    })
    feats["qty_vs_material_median"] = feats["qty_vs_material_median"].fillna(1.0)
    return feats[FEATURES]


def history_to_targets(history: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"supplier": history["supplier"], "material": history["material"],
                         "order_date": history["order_date"], "quantity": history["quantity_ordered"],
                         "unit_price": history["unit_price"], "promised_lead_days": history["promised_lead_days"]})


def _scores(y_true, pred, proba) -> dict:
    auc = roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan")
    return {"Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "F1": f1_score(y_true, pred, zero_division=0), "ROC-AUC": auc}


def _train_target(name: str, X, y, split: int) -> dict:
    """Pick the best candidate by time-ordered cross-validation, report held-out test results."""
    X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    if y.nunique() < 2:
        raise ValueError(f"The history has no examples of both outcomes for “{name}”, so a model cannot be trained.")
    cv = {}
    for model_name, factory in CANDIDATES.items():
        if model_name == "Baseline":
            continue
        folds = []
        for a, b in TimeSeriesSplit(n_splits=3).split(X_tr):
            if y_tr.iloc[a].nunique() < 2 or y_tr.iloc[b].nunique() < 2:
                continue
            m = factory().fit(X_tr.iloc[a], y_tr.iloc[a])
            folds.append(roc_auc_score(y_tr.iloc[b], m.predict_proba(X_tr.iloc[b])[:, 1]))
        cv[model_name] = float(np.mean(folds)) if folds else float("nan")
    valid = {k: v for k, v in cv.items() if not math.isnan(v)}
    chosen = max(valid, key=valid.get) if valid else "Logistic Regression"

    rows, fitted = [], {}
    for model_name, factory in CANDIDATES.items():
        m = factory().fit(X_tr, y_tr)
        fitted[model_name] = m
        proba = m.predict_proba(X_te)[:, 1] if len(y_tr.unique()) > 1 else np.zeros(len(X_te))
        pred = (proba >= MEDIUM_RISK).astype(int)
        rows.append({"Model": model_name, **_scores(y_te, pred, proba), "CV ROC-AUC (train)": cv.get(model_name, float("nan"))})
    best_proba = fitted[chosen].predict_proba(X_te)[:, 1]
    confusion = confusion_matrix(y_te, (best_proba >= MEDIUM_RISK).astype(int), labels=[0, 1]).tolist()
    try:
        imp = permutation_importance(fitted[chosen], X_te, y_te, scoring="roc_auc", n_repeats=5,
                                     random_state=RANDOM_STATE, n_jobs=1)
        importance = sorted(zip(X.columns, imp.importances_mean), key=lambda p: -p[1])
    except Exception:
        importance = []
    return {"chosen": chosen, "model": CANDIDATES[chosen]().fit(X, y), "comparison": rows,
            "confusion": confusion, "importance": importance, "positive_rate": float(y.mean())}


def train_models(history: pd.DataFrame) -> dict:
    """Train the 'late' and 'incomplete' classifiers. Raises ValueError if the history is unusable."""
    if len(history) < MIN_HISTORY_ROWS:
        raise ValueError(f"At least {MIN_HISTORY_ROWS} historical orders are needed; found {len(history)}.")
    h = prepare_history(history).sort_values("order_date", kind="stable").reset_index(drop=True)
    X = compute_features(history_to_targets(h), h)
    split = int(len(h) * 0.75)
    targets = {"late": _train_target("late delivery", X, h["is_late"], split),
               "incomplete": _train_target("incomplete delivery", X, h["is_incomplete"], split)}
    return {"rows": len(h), "train_rows": split, "test_rows": len(h) - split,
            "baseline_values": X.median().to_dict(), "targets": targets}


def predict_orders(bundle: dict, targets: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (targets + probabilities, feature table)."""
    feats = compute_features(targets, history)
    out = targets.reset_index(drop=True).copy()
    out["p_late"] = bundle["targets"]["late"]["model"].predict_proba(feats)[:, 1]
    out["p_incomplete"] = bundle["targets"]["incomplete"]["model"].predict_proba(feats)[:, 1]
    out["p_risk"] = 1 - (1 - out["p_late"]) * (1 - out["p_incomplete"])     # assumes the two are independent
    out["risk_level"] = out["p_risk"].map(risk_level)
    return out, feats


def format_value(feature: str, value: float) -> str:
    if feature in ("supplier_late_rate", "supplier_incomplete_rate", "material_late_rate"):
        return f"{value:.0%}"
    if feature in ("log_quantity", "log_order_value"):
        return f"{math.expm1(value):,.0f}"
    if feature == "qty_vs_material_median":
        return f"{value:.1f}x"
    if feature == "supplier_avg_delay_days":
        return f"{value:.1f}"
    return f"{value:.0f}"


def explain_order(bundle: dict, feature_row: pd.Series, target: str, top: int = 4) -> list[dict]:
    """Which features pushed this order's probability up or down?

    Each feature is swapped for its typical (median) value in turn; the change in the
    predicted probability is that feature's effect. Works for any scikit-learn model.
    """
    model = bundle["targets"][target]["model"]
    base = bundle["baseline_values"]
    row = feature_row[FEATURES].astype(float)
    variants = pd.DataFrame([row] * (len(FEATURES) + 1)).reset_index(drop=True)
    for i, f in enumerate(FEATURES):
        variants.loc[i, f] = base[f]
    proba = model.predict_proba(variants[FEATURES])[:, 1]
    full = proba[-1]
    effects = sorted(((f, full - proba[i]) for i, f in enumerate(FEATURES)), key=lambda p: -abs(p[1]))[:top]
    label = "late" if target == "late" else "incomplete"
    out = []
    for f, delta in effects:
        direction = "raises" if delta > 0 else "lowers"
        out.append({"feature": f, "label": FEATURE_LABELS[f], "value": format_value(f, row[f]),
                    "effect": float(delta),
                    "text": f"{FEATURE_LABELS[f]} ({format_value(f, row[f])}) {direction} {label} risk by "
                            f"{abs(delta) * 100:.0f} points"})
    return out


def supplier_outcome_stats(history: pd.DataFrame) -> tuple[dict, float, float]:
    """Typical delay (when late) and typical shortfall (when incomplete) per supplier, plus overall defaults."""
    h = prepare_history(history)
    late, inc = h[h["is_late"] == 1], h[h["is_incomplete"] == 1]
    overall_delay = float(late["delay_days"].mean()) if len(late) else 3.0
    overall_short = float(inc["shortfall_ratio"].mean()) if len(inc) else 0.2
    per = {}
    for supplier in h["supplier"].unique():
        l, i = late[late["supplier"] == supplier], inc[inc["supplier"] == supplier]
        per[supplier] = {"avg_delay_days": float(l["delay_days"].mean()) if len(l) >= 3 else overall_delay,
                         "avg_shortfall": float(i["shortfall_ratio"].mean()) if len(i) >= 3 else overall_short}
    return per, overall_delay, overall_short
