"""SQLite procurement database.

Documents are stored in the same shape the extractor produces, so the matching
engine can work on database records or freshly extracted JSON alike.
Money is stored as text (Decimal-safe); queries are always parameterised.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = os.environ.get("PROCUREMENT_DB", "data/procurement.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_number TEXT PRIMARY KEY, supplier TEXT, order_date TEXT, delivery_date TEXT,
    currency TEXT, subtotal TEXT, tax TEXT, total TEXT, source_filename TEXT);
CREATE TABLE IF NOT EXISTS po_items (
    id INTEGER PRIMARY KEY, po_number TEXT NOT NULL,
    name TEXT, quantity TEXT, unit_price TEXT, line_total TEXT);
CREATE TABLE IF NOT EXISTS grns (
    grn_number TEXT PRIMARY KEY, po_number TEXT, supplier TEXT, receipt_date TEXT,
    source_filename TEXT);
CREATE TABLE IF NOT EXISTS grn_items (
    id INTEGER PRIMARY KEY, grn_number TEXT NOT NULL,
    name TEXT, quantity TEXT, unit_price TEXT, line_total TEXT);
CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY, invoice_number TEXT, po_number TEXT, supplier TEXT,
    invoice_date TEXT, currency TEXT, subtotal TEXT, tax TEXT, total TEXT,
    source_filename TEXT, file_sha256 TEXT, saved_at TEXT);
CREATE TABLE IF NOT EXISTS invoice_items (
    id INTEGER PRIMARY KEY, invoice_id INTEGER NOT NULL,
    name TEXT, quantity TEXT, unit_price TEXT, line_total TEXT);
CREATE TABLE IF NOT EXISTS history_orders (
    po_number TEXT, supplier TEXT, material TEXT, order_date TEXT,
    quantity_ordered REAL, unit_price REAL, promised_lead_days INTEGER,
    actual_lead_days INTEGER, quantity_received REAL);
CREATE TABLE IF NOT EXISTS inventory (
    material TEXT PRIMARY KEY, stock REAL, daily_demand REAL);
"""

HISTORY_COLUMNS = ["po_number", "supplier", "material", "order_date", "quantity_ordered",
                   "unit_price", "promised_lead_days", "actual_lead_days", "quantity_received"]
INVENTORY_COLUMNS = ["material", "stock", "daily_demand"]
ALL_TABLES = ["po_items", "purchase_orders", "grn_items", "grns", "invoice_items", "invoices",
              "history_orders", "inventory", "meta"]


@contextmanager
def get_conn(path: str | None = None):
    """Open the database (creating it if needed); commit on success."""
    path = path or DEFAULT_DB_PATH
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _dec(value) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _txt(value) -> str | None:
    return None if value is None else str(value)


# ------------------------------------------------------------------ saving
def _insert_items(conn, table: str, key_col: str, key, items: list[dict]) -> None:
    for item in items or []:
        conn.execute(
            f"INSERT INTO {table} ({key_col}, name, quantity, unit_price, line_total) VALUES (?,?,?,?,?)",
            (key, item.get("name"), _txt(item.get("quantity")), _txt(item.get("unit_price")),
             _txt(item.get("line_total"))))


def save_document(conn, data: dict, file_sha256: str | None = None) -> tuple[str, str]:
    """Store a reviewed document. Returns (kind, identifier). Raises ValueError if unusable."""
    doc_type = data.get("document_type")
    number = (data.get("document_id") or "").strip() if data.get("document_id") else ""
    if not number:
        raise ValueError("Document ID is missing. Add it in the JSON before saving.")
    items = data.get("items") or []

    if doc_type == "Purchase Order":
        conn.execute("DELETE FROM po_items WHERE po_number=?", (number,))
        conn.execute("DELETE FROM purchase_orders WHERE po_number=?", (number,))
        conn.execute(
            "INSERT INTO purchase_orders VALUES (?,?,?,?,?,?,?,?,?)",
            (number, data.get("vendor"), data.get("document_date"), data.get("delivery_date"),
             data.get("currency"), _txt(data.get("subtotal")), _txt(data.get("tax")),
             _txt(data.get("total")), data.get("source_filename")))
        _insert_items(conn, "po_items", "po_number", number, items)
        return "Purchase Order", number

    if doc_type == "Delivery Receipt / GRN":
        if not data.get("po_reference"):
            raise ValueError("PO reference is missing. Add it so the receipt can be linked to its order.")
        conn.execute("DELETE FROM grn_items WHERE grn_number=?", (number,))
        conn.execute("DELETE FROM grns WHERE grn_number=?", (number,))
        conn.execute(
            "INSERT INTO grns VALUES (?,?,?,?,?)",
            (number, data.get("po_reference"), data.get("vendor"), data.get("document_date"),
             data.get("source_filename")))
        _insert_items(conn, "grn_items", "grn_number", number, items)
        return "Delivery Receipt / GRN", number

    if doc_type == "Invoice":
        cur = conn.execute(
            "INSERT INTO invoices (invoice_number, po_number, supplier, invoice_date, currency, subtotal,"
            " tax, total, source_filename, file_sha256, saved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (number, data.get("po_reference"), data.get("vendor"), data.get("document_date"),
             data.get("currency"), _txt(data.get("subtotal")), _txt(data.get("tax")),
             _txt(data.get("total")), data.get("source_filename"), file_sha256,
             datetime.now().isoformat(timespec="seconds")))
        _insert_items(conn, "invoice_items", "invoice_id", cur.lastrowid, items)
        return "Invoice", str(cur.lastrowid)

    raise ValueError(f"Unknown document type: {doc_type!r}")


# ----------------------------------------------------------------- reading
def _items_for(conn, table: str, key_col: str, key) -> list[dict]:
    rows = conn.execute(
        f"SELECT name, quantity, unit_price, line_total FROM {table} WHERE {key_col}=? ORDER BY id", (key,))
    return [dict(r) for r in rows]


def _blank(doc_type: str) -> dict:
    from .extract_fields import empty_document
    return empty_document(doc_type)


def get_po(conn, po_number: str) -> dict | None:
    row = conn.execute("SELECT * FROM purchase_orders WHERE po_number=?", (po_number,)).fetchone()
    if row is None:
        return None
    data = _blank("Purchase Order")
    data.update(document_id=row["po_number"], vendor=row["supplier"], document_date=row["order_date"],
                delivery_date=row["delivery_date"], currency=row["currency"], subtotal=row["subtotal"],
                tax=row["tax"], total=row["total"], source_filename=row["source_filename"],
                items=_items_for(conn, "po_items", "po_number", po_number))
    return data


def list_po_numbers(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT po_number FROM purchase_orders ORDER BY po_number")]


def get_grns_for_po(conn, po_number: str) -> list[dict]:
    out = []
    for row in conn.execute("SELECT * FROM grns WHERE po_number=? ORDER BY grn_number", (po_number,)):
        data = _blank("Delivery Receipt / GRN")
        data.update(document_id=row["grn_number"], po_reference=row["po_number"], vendor=row["supplier"],
                    document_date=row["receipt_date"], source_filename=row["source_filename"],
                    items=_items_for(conn, "grn_items", "grn_number", row["grn_number"]))
        out.append(data)
    return out


def _invoice_from_row(conn, row) -> dict:
    data = _blank("Invoice")
    data.update(document_id=row["invoice_number"], po_reference=row["po_number"], vendor=row["supplier"],
                document_date=row["invoice_date"], currency=row["currency"], subtotal=row["subtotal"],
                tax=row["tax"], total=row["total"], source_filename=row["source_filename"],
                items=_items_for(conn, "invoice_items", "invoice_id", row["id"]))
    data["id"] = row["id"]                       # extra keys used for duplicate checks
    data["file_sha256"] = row["file_sha256"]
    return data


def get_invoice(conn, invoice_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM invoices WHERE id=?", (invoice_id,)).fetchone()
    return None if row is None else _invoice_from_row(conn, row)


def list_invoices(conn, po_number: str | None = None) -> list[dict]:
    if po_number is None:
        rows = conn.execute("SELECT * FROM invoices ORDER BY id")
    else:
        rows = conn.execute("SELECT * FROM invoices WHERE po_number=? ORDER BY id", (po_number,))
    return [_invoice_from_row(conn, r) for r in rows.fetchall()]


def all_invoice_headers(conn) -> list[dict]:
    """Light records (no items) for duplicate checks."""
    return [dict(r) for r in conn.execute(
        "SELECT id, invoice_number AS document_id, supplier AS vendor, invoice_date AS document_date,"
        " total, file_sha256 FROM invoices ORDER BY id")]


def get_open_po_lines(conn) -> pd.DataFrame:
    """PO lines that have not been fully received yet (received quantity summed over all GRNs)."""
    rows = []
    for po_number in list_po_numbers(conn):
        po = get_po(conn, po_number)
        received: dict[str, Decimal] = {}
        for grn in get_grns_for_po(conn, po_number):
            for item in grn["items"]:
                key = " ".join((item["name"] or "").lower().split())
                received[key] = received.get(key, Decimal(0)) + (_dec(item["quantity"]) or Decimal(0))
        for item in po["items"]:
            key = " ".join((item["name"] or "").lower().split())
            ordered = _dec(item["quantity"]) or Decimal(0)
            got = received.get(key, Decimal(0))
            remaining = ordered - got
            if remaining > 0:
                rows.append({
                    "po_number": po_number, "supplier": po["vendor"], "material": item["name"],
                    "order_date": po["document_date"], "promised_date": po["delivery_date"],
                    "quantity_ordered": float(ordered), "quantity_received": float(got),
                    "remaining_qty": float(remaining),
                    "unit_price": float(_dec(item["unit_price"]) or 0)})
    return pd.DataFrame(rows, columns=[
        "po_number", "supplier", "material", "order_date", "promised_date", "quantity_ordered",
        "quantity_received", "remaining_qty", "unit_price"])


# ------------------------------------------------------- history/inventory
def save_history(conn, df: pd.DataFrame, replace: bool = True) -> int:
    missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("History file is missing columns: " + ", ".join(missing))
    out = df[HISTORY_COLUMNS].copy()
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    out = out.dropna(subset=["supplier", "material", "order_date", "quantity_ordered",
                             "promised_lead_days", "actual_lead_days", "quantity_received"])
    out["order_date"] = out["order_date"].dt.strftime("%Y-%m-%d")
    if replace:
        conn.execute("DELETE FROM history_orders")
    out.to_sql("history_orders", conn, if_exists="append", index=False)
    return len(out)


def load_history(conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM history_orders", conn)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def save_inventory(conn, df: pd.DataFrame, replace: bool = True) -> int:
    missing = [c for c in INVENTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Inventory file is missing columns: " + ", ".join(missing))
    out = df[INVENTORY_COLUMNS].copy()
    out["stock"] = pd.to_numeric(out["stock"], errors="coerce")
    out["daily_demand"] = pd.to_numeric(out["daily_demand"], errors="coerce")
    out = out.dropna(subset=["material"]).drop_duplicates("material", keep="last")
    if replace:
        conn.execute("DELETE FROM inventory")
    out.to_sql("inventory", conn, if_exists="append", index=False)
    return len(out)


def load_inventory(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM inventory", conn)


# ------------------------------------------------------------ match results
def set_meta(conn, key: str, value) -> None:
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))


def get_meta(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return default if row is None else json.loads(row["value"])


def counts(conn) -> dict:
    names = {"purchase_orders": "Purchase orders", "grns": "Receipts (GRN)", "invoices": "Invoices",
             "history_orders": "Historical orders", "inventory": "Inventory items"}
    return {label: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t, label in names.items()}


def clear_all(conn) -> None:
    for table in ALL_TABLES:
        conn.execute(f"DELETE FROM {table}")
