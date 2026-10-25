from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import demo_data
from src.risk_model import (FEATURES, MIN_HISTORY_ROWS, compute_features, explain_order, history_to_targets,
                            predict_orders, prepare_history, risk_level, supplier_outcome_stats, train_models)

TODAY = date(2026, 10, 22)


@pytest.fixture(scope="module")
def history():
    return demo_data.generate_history(TODAY, n=600, seed=3)


@pytest.fixture(scope="module")
def bundle(history):
    return train_models(history)


def test_labels_are_built_from_outcomes(history):
    h = prepare_history(history)
    row = h.iloc[0]
    assert row["is_late"] == int(row["actual_lead_days"] > row["promised_lead_days"])
    assert (h["shortfall_ratio"] >= 0).all()


def test_features_do_not_use_future_outcomes(history):
    """Changing the outcome of an order that finished AFTER the scored order must not change its features."""
    target = pd.DataFrame([{"supplier": history["supplier"].iloc[0], "material": history["material"].iloc[0],
                            "order_date": pd.Timestamp("2025-06-01"), "quantity": 100, "unit_price": 10.0,
                            "promised_lead_days": 10}])
    base = compute_features(target, history)
    h2 = prepare_history(history)
    future = h2["receipt_day"] > (pd.Timestamp("2025-06-01") - pd.Timestamp("1970-01-01")).days
    tampered = history.copy()
    tampered.loc[future, "actual_lead_days"] = tampered.loc[future, "actual_lead_days"] + 50
    tampered.loc[future, "quantity_received"] = 0
    pd.testing.assert_frame_equal(base, compute_features(target, tampered))


def test_feature_table_shape_and_no_nans(history):
    feats = compute_features(history_to_targets(history), history)
    assert list(feats.columns) == FEATURES and len(feats) == len(history)
    assert not feats.isna().any().any()


def test_unseen_supplier_gets_prior_not_error(history):
    t = pd.DataFrame([{"supplier": "Brand New Co", "material": "Unknown Thing", "order_date": pd.Timestamp("2026-10-01"),
                       "quantity": 5, "unit_price": 1.0, "promised_lead_days": 7}])
    f = compute_features(t, history)
    assert f["supplier_order_count"].iloc[0] == 0 and 0 < f["supplier_late_rate"].iloc[0] < 1
    assert f["qty_vs_material_median"].iloc[0] == 1.0


def test_training_results_are_reasonable(bundle):
    for key in ("late", "incomplete"):
        t = bundle["targets"][key]
        assert t["chosen"] in ("Logistic Regression", "Decision Tree", "Random Forest")
        names = [r["Model"] for r in t["comparison"]]
        assert names[0] == "Baseline" and len(names) == 4
        assert all(0 <= r["ROC-AUC"] <= 1 for r in t["comparison"])
        assert sum(map(sum, t["confusion"])) == bundle["test_rows"]
    # the real models should beat the do-nothing baseline on the late target
    comp = {r["Model"]: r["ROC-AUC"] for r in bundle["targets"]["late"]["comparison"]}
    assert max(comp["Logistic Regression"], comp["Random Forest"]) > comp["Baseline"] + 0.1


def test_predictions_and_explanations(bundle, history):
    t = history_to_targets(history.tail(5))
    out, feats = predict_orders(bundle, t, history)
    assert out[["p_late", "p_incomplete", "p_risk"]].apply(lambda c: c.between(0, 1).all()).all()
    assert (out["p_risk"] >= out[["p_late", "p_incomplete"]].max(axis=1) - 1e-9).all()
    items = explain_order(bundle, feats.iloc[0], "late", top=3)
    assert len(items) == 3 and all({"label", "effect", "text"} <= set(i) for i in items)
    assert abs(items[0]["effect"]) >= abs(items[-1]["effect"])


def test_training_is_repeatable(history):
    a, b = train_models(history), train_models(history)
    t = history_to_targets(history.tail(10))
    pa, _ = predict_orders(a, t, history)
    pb, _ = predict_orders(b, t, history)
    np.testing.assert_allclose(pa["p_late"], pb["p_late"])


def test_too_little_or_one_sided_history_is_rejected(history):
    with pytest.raises(ValueError):
        train_models(history.head(MIN_HISTORY_ROWS - 1))
    never_late = history.copy()
    never_late["actual_lead_days"] = never_late["promised_lead_days"]
    with pytest.raises(ValueError):
        train_models(never_late)


def test_risk_levels():
    assert (risk_level(0.8), risk_level(0.5), risk_level(0.3), risk_level(0.29)) == ("High", "High", "Medium", "Low")


def test_supplier_outcome_stats_have_defaults(history):
    per, delay, short = supplier_outcome_stats(history)
    assert delay > 0 and 0 < short < 1 and set(per) == set(history["supplier"])
