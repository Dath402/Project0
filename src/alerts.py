"""Turn match results, delivery risks and shortage risks into one ranked alert list."""
import pandas as pd

BASE_SCORE = {"High": 80, "Medium": 50, "Low": 20}
ACTIONS = {
    "DUPLICATE_INVOICE": "Do not pay. Confirm with the supplier and check whether the earlier invoice was paid.",
    "POSSIBLE_DUPLICATE": "Check whether this repeats an earlier invoice before approving.",
    "GRN_NOT_FOUND": "Hold payment until the warehouse confirms the goods arrived.",
    "PO_NOT_FOUND": "Find the purchase order or reject the invoice.",
    "QTY_EXCEEDS_RECEIPT": "Pay only for the received quantity or ask the supplier to correct the invoice.",
    "QTY_EXCEEDS_PO": "Ask the supplier why more than the ordered quantity was billed.",
    "PRICE_HIGHER_THAN_PO": "Ask the supplier for a corrected invoice at the agreed price.",
    "ITEM_NOT_ON_PO": "Ask the supplier to explain the item or remove it from the invoice.",
    "ITEM_NOT_RECEIVED": "Check receiving records for this item before approving.",
    "SUPPLIER_MISMATCH": "Verify the supplier identity before paying.",
}


def invoice_alerts(match_results: list[dict]) -> list[dict]:
    """match_results: [{'invoice': 'INV-1', 'po': 'PO-1', 'result': <three_way_match output>}]"""
    out = []
    for m in match_results:
        for d in m["result"]["discrepancies"]:
            sev = d["severity"].capitalize()
            out.append({"score": BASE_SCORE[sev] + (5 if "DUPLICATE" in d["code"] else 0),
                        "category": "Invoice", "severity": sev,
                        "title": f"Invoice {m['invoice']} (PO {m['po']}): {d['message']}",
                        "action": ACTIONS.get(d["code"], "Review this difference with the supplier."),
                        "ref": m["invoice"]})
    return out


def delivery_alerts(risk: pd.DataFrame, reasons: dict, as_of) -> list[dict]:
    """risk needs: po_number, supplier, material, promised_date, p_late, p_incomplete, p_risk, risk_level."""
    out = []
    for r in risk.itertuples():
        overdue = pd.notna(r.promised_date) and pd.to_datetime(r.promised_date).date() < as_of
        if r.risk_level == "Low" and not overdue:
            continue
        sev = "High" if overdue or r.risk_level == "High" else "Medium"
        why = reasons.get((r.po_number, r.material), "")
        title = (f"PO {r.po_number}, {r.material} from {r.supplier}: "
                 + ("delivery is overdue. " if overdue else "")
                 + f"risk {r.p_risk:.0%} (late {r.p_late:.0%}, incomplete {r.p_incomplete:.0%}).")
        out.append({"score": BASE_SCORE[sev] + 10 * float(r.p_risk), "category": "Delivery", "severity": sev,
                    "title": title + (f" {why}" if why else ""),
                    "action": f"Contact {r.supplier} to confirm the delivery date and quantity for {r.po_number}.",
                    "ref": r.po_number})
    return out


def shortage_alerts(shortage: pd.DataFrame) -> list[dict]:
    out = []
    for r in shortage.itertuples():
        if r.risk not in ("High", "Medium"):
            continue
        out.append({"score": BASE_SCORE[r.risk] + 5, "category": "Shortage", "severity": r.risk,
                    "title": f"{r.material}: {r.reason}",
                    "action": "Expedite open orders, find an alternative supplier or reduce planned use.",
                    "ref": r.material})
    return out


def build_alerts(match_results: list[dict], risk: pd.DataFrame, reasons: dict, shortage: pd.DataFrame, as_of) -> pd.DataFrame:
    alerts = invoice_alerts(match_results) + delivery_alerts(risk, reasons, as_of) + shortage_alerts(shortage)
    df = pd.DataFrame(alerts, columns=["score", "category", "severity", "title", "action", "ref"])
    return df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
