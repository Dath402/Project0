"""Rule-based field extraction.

Turns plain text (from a PDF or OCR) into one fixed JSON-friendly schema.
It is a starting point, NOT a universal invoice parser: every value must be
reviewed by a person. Anything we cannot find or are unsure about is None.
"""
import re
from datetime import date
from decimal import Decimal, InvalidOperation

SCHEMA_KEYS = (
    "document_type", "document_id", "po_reference", "vendor", "document_date",
    "delivery_date", "currency", "items", "subtotal", "tax", "total", "source_filename",
)
ITEM_KEYS = ("name", "quantity", "unit_price", "line_total")


def empty_document(document_type: str | None = None) -> dict:
    """Blank record with every schema key present."""
    return {
        "document_type": document_type, "document_id": None, "po_reference": None,
        "vendor": None, "document_date": None, "delivery_date": None,
        "currency": None, "items": [],
        "subtotal": None, "tax": None, "total": None, "source_filename": None,
    }


# ----------------------------------------------------------------- numbers
_CURRENCY_NOISE = re.compile(r"(?i)(?:₹|\$|€|£|\bINR\b|\bUSD\b|\bEUR\b|\bGBP\b|\bRs\.?)")
_PLAIN_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_GROUPED_NUMBER = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?")  # 1,234.50 and 1,23,456.00


def parse_amount(value) -> Decimal | None:
    """Parse '1,234.50', '₹ 5000', 'Rs. 1,23,456.00' to Decimal.

    Returns None when the text is not clearly a number (for example '1.234,56',
    where we cannot know whether the comma is a decimal point).
    """
    if value is None:
        return None
    cleaned = _CURRENCY_NOISE.sub("", str(value)).strip()
    if _PLAIN_NUMBER.fullmatch(cleaned) or _GROUPED_NUMBER.fullmatch(cleaned):
        try:
            return Decimal(cleaned.replace(",", ""))
        except InvalidOperation:
            return None
    return None


def _amount_str(value) -> str | None:
    amount = parse_amount(value)
    return None if amount is None else str(amount)


# ------------------------------------------------------------------- dates
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _make_date(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(text: str) -> tuple[str | None, bool]:
    """Return (ISO date or None, is_ambiguous).

    Numeric dates like 03/04/2026 can be day-first or month-first. We read them
    day-first (common in India) and flag them as ambiguous so a warning is shown.
    """
    text = text.strip()
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", text)          # 2026-10-20
    if m:
        return _make_date(int(m[1]), int(m[2]), int(m[3])), False
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b", text)          # 20/10/2026
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        ambiguous = a <= 12 and b <= 12 and a != b
        return _make_date(y, b, a), ambiguous
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s-]+([A-Za-z]{3,9})\.?,?[\s-]+(\d{4})\b", text)  # 20 Oct 2026
    if m and m[2][:3].lower() in _MONTHS:
        return _make_date(int(m[3]), _MONTHS[m[2][:3].lower()], int(m[1])), False
    m = re.search(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", text)  # Oct 20, 2026
    if m and m[1][:3].lower() in _MONTHS:
        return _make_date(int(m[3]), _MONTHS[m[1][:3].lower()], int(m[2])), False
    return None, False


# ---------------------------------------------------------- labelled values
def _label_value(patterns: list[str], text: str) -> str | None:
    """First regex hit; the value stops at a column gap (2+ spaces or a pipe)."""
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            value = re.split(r"\s{2,}|\s\|\s", m.group(1).strip())[0].strip(" \t:#-")
            if value:
                return value
    return None


_ID_TOKEN = r"([A-Z0-9][A-Z0-9/_.\-]*)"
_SEP = r"\s*[:#\-]?\s*"
ID_PATTERNS = {
    "Invoice": [
        rf"\b(?:tax\s+)?invoice\s*(?:number|no\.?|num|#|id){_SEP}{_ID_TOKEN}",
        rf"\bbill\s*(?:number|no\.?|#){_SEP}{_ID_TOKEN}",
    ],
    "Purchase Order": [
        rf"\bpurchase\s*order\s*(?:number|no\.?|num|#|id)?{_SEP}{_ID_TOKEN}",
        rf"\bP\.?O\.?\s*(?:number|no\.?|num|#|id){_SEP}{_ID_TOKEN}",
        rf"\border\s*(?:number|no\.?|#){_SEP}{_ID_TOKEN}",
    ],
    "Delivery Receipt / GRN": [
        rf"\bGRN\s*(?:number|no\.?|num|#|id)?{_SEP}{_ID_TOKEN}",
        rf"\bgoods\s*receipt(?:\s*note)?\s*(?:number|no\.?|#)?{_SEP}{_ID_TOKEN}",
        rf"\bdelivery\s*(?:receipt|note|challan)\s*(?:number|no\.?|num|#|id)?{_SEP}{_ID_TOKEN}",
        rf"\breceipt\s*(?:number|no\.?|#){_SEP}{_ID_TOKEN}",
    ],
}


def _find_document_id(text: str, document_type: str) -> str | None:
    """Try labels for the chosen type first, then the other types."""
    ordered = list(ID_PATTERNS.get(document_type, []))
    for key, pats in ID_PATTERNS.items():
        if key != document_type:
            ordered.extend(pats)
    for pattern in ordered:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            token = m.group(1).strip("-_./")
            if re.search(r"\d", token):          # real IDs contain a digit; skips words like "Date"
                return token
    return None


PO_REF_PATTERNS = [
    rf"\bpurchase\s*order\s*(?:number|no\.?|num|#|id|ref(?:erence)?)?{_SEP}{_ID_TOKEN}",
    rf"\bP\.?O\.?\s*(?:number|no\.?|num|#|id|ref(?:erence)?){_SEP}{_ID_TOKEN}",
    rf"\border\s*(?:number|no\.?|#|ref(?:erence)?){_SEP}{_ID_TOKEN}",
]


def _find_po_reference(text: str, document_type: str) -> str | None:
    """The purchase order an invoice or GRN refers to (a PO's own number is its document_id)."""
    if document_type == "Purchase Order":
        return None
    for pattern in PO_REF_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            token = m.group(1).strip("-_./")
            if re.search(r"\d", token):
                return token
    return None


VENDOR_PATTERNS = [
    r"^[ \t]*(?:vendor|supplier|seller|sold\s*by|billed\s*by|issued\s*by)(?:\s*name)?[ \t]*[:\-][ \t]*(.+)$",
]
DATE_LABELS = [
    r"(?:invoice|order|po|grn|receipt|delivery|document|bill)\s*date",
    r"\bdated?\b",
]


def _find_date(text: str) -> tuple[str | None, bool]:
    for label in DATE_LABELS:
        for m in re.finditer(rf"{label}\s*[:\-]?\s*([^\n]+)", text, flags=re.IGNORECASE):
            iso, ambiguous = parse_date(m.group(1)[:30])
            if iso:
                return iso, ambiguous
    return None, False


DELIVERY_DATE_LABELS = [
    r"(?:expected|promised|requested|required)\s*(?:delivery\s*)?date",
    r"delivery\s*(?:date|due)",
    r"due\s*date",
    r"deliver\s*by",
]


def _find_delivery_date(text: str) -> str | None:
    for label in DELIVERY_DATE_LABELS:
        for m in re.finditer(rf"{label}\s*[:\-]?\s*([^\n]+)", text, flags=re.IGNORECASE):
            iso, _ = parse_date(m.group(1)[:30])
            if iso:
                return iso
    return None


_SYMBOL_TO_CODE = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}


def _find_currency(text: str) -> tuple[str | None, bool]:
    """Return (code, conflict). Conflict means several currencies appear."""
    m = re.search(r"currency\s*[:\-]?\s*(INR|USD|EUR|GBP)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), False
    found = set()
    for symbol, code in _SYMBOL_TO_CODE.items():
        if symbol in text:
            found.add(code)
    for code in ("INR", "USD", "EUR", "GBP"):
        if re.search(rf"\b{code}\b", text):
            found.add(code)
    if re.search(r"\bRs\.?(?=\s*\d)", text):
        found.add("INR")
    if len(found) == 1:
        return found.pop(), False
    return None, len(found) > 1


# ------------------------------------------------------------------ totals
_MONEY = r"(?:₹|\$|€|£|INR|USD|EUR|GBP|Rs\.?)?\s*(-?[\d,]+(?:\.\d+)?)"
_TAIL = rf"\s*[:\-|]?\s*{_MONEY}\s*$"


def _amount_after_label(label: str, lines: list[str]) -> str | None:
    """Last line that is 'label ... amount' (summary lines sit at the bottom)."""
    found = None
    for line in lines:
        m = re.match(rf"^\s*(?:{label}){_TAIL}", line, flags=re.IGNORECASE)
        if m and parse_amount(m.group(1)) is not None:
            found = _amount_str(m.group(1))
    return found


SUBTOTAL_LABEL = r"sub\s*-?\s*total|total\s+before\s+tax"
TAX_LABEL = r"(?:total\s+)?(?:tax|gst|vat|igst)(?:\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?)?"
# Plain "total" must not be followed by qty/items words (e.g. "Total Qty: 15").
GRAND_LABELS = [
    (r"grand\s*total|invoice\s*total|total\s*amount|total\s*payable|net\s*payable", False),
    (r"total\s*due|total(?!\s*(?:qty|quantity|items?|units?|pcs|tax|gst|vat))", False),
    (r"amount\s*due|balance\s*due", True),   # may include a previous balance
]


def _extract_summary(text: str) -> tuple[dict, list[str]]:
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    warnings: list[str] = []
    subtotal = _amount_after_label(SUBTOTAL_LABEL, lines)
    tax = _amount_after_label(TAX_LABEL, lines)
    if tax is None:   # split taxes: CGST + SGST
        cgst = _amount_after_label(r"cgst(?:\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?)?", lines)
        sgst = _amount_after_label(r"sgst(?:\s*\(?\s*\d+(?:\.\d+)?\s*%\s*\)?)?", lines)
        if cgst and sgst:
            tax = str(Decimal(cgst) + Decimal(sgst))
            warnings.append("Tax was calculated as CGST + SGST because no single tax line was found.")
    total = None
    for label, may_include_balance in GRAND_LABELS:
        total = _amount_after_label(label, lines)
        if total is not None:
            if may_include_balance:
                warnings.append(
                    "Total was taken from an 'Amount/Balance Due' line, which may include a "
                    "previous balance. Please check it.")
            break
    return {"subtotal": subtotal, "tax": tax, "total": total}, warnings


# ------------------------------------------------------------- line items
_HEADER_WORDS = re.compile(
    r"\b(?:description|item|particulars|product|qty|quantity|unit\s*price|price|rate|amount|total)\b", re.I)
_SUMMARY_START = re.compile(
    r"^\s*(?:sub\s*-?\s*total|total|grand\s*total|tax|gst|cgst|sgst|igst|vat|discount|shipping|"
    r"freight|round\s*off|balance|previous|amount\s*due|amount\s*in\s*words|net\s*payable)\b", re.I)

_QTY = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
_AMT = r"(?:₹|\$|€|£|Rs\.?)?\s*\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?|(?:₹|\$|€|£|Rs\.?)?\s*\d+(?:\.\d{1,2})?"
_WS_ROW = re.compile(
    rf"^\s*(?P<name>\S.*?)\s{{2,}}(?P<qty>{_QTY})\s{{2,}}(?P<price>{_AMT})\s{{2,}}(?P<total>{_AMT})\s*$")


def _is_header(norm: str) -> bool:
    """A header line has at least two column words and no real numbers."""
    return len(_HEADER_WORDS.findall(norm)) >= 2 and not re.search(r"\d", norm)


def _make_item(name, qty, price, total) -> dict | None:
    """Build an item only when all numbers parse and the name has a letter."""
    q, p, t = parse_amount(qty), parse_amount(price), parse_amount(total)
    if None in (q, p, t) or not re.search(r"[A-Za-z]", name):
        return None
    return {"name": name.strip(), "quantity": str(q), "unit_price": str(p), "line_total": str(t)}


_WS_QTY_ONLY = re.compile(rf"^\s*(?P<name>\S.*?)\s{{2,}}(?P<qty>{_QTY})\s*$")


def _make_qty_item(name, qty) -> dict | None:
    """GRN row with only a description and a received quantity."""
    q = parse_amount(qty)
    if q is None or not re.search(r"[A-Za-z]", name):
        return None
    return {"name": name.strip(), "quantity": str(q), "unit_price": None, "line_total": None}


def extract_items(text: str, allow_quantity_only: bool = False) -> tuple[list[dict], list[str]]:
    """Find rows of: description, quantity, unit price, line total.

    Two kinds of row are understood:
      * pipe rows      -> Name | 2 | 10.00 | 20.00     (accepted anywhere)
      * aligned rows   -> Name   2   10.00   20.00     (only after a header line)
    The original line is kept for the aligned parser, because the 2+ space gaps
    ARE the column separators. A whitespace-normalised copy is used only for
    checking headers and summary lines.
    """
    items: list[dict] = []
    warnings: list[str] = []
    in_table = False
    unparsed: list[str] = []

    for original in text.splitlines():
        line = original.rstrip()
        norm = re.sub(r"\s+", " ", line).strip()
        if len(norm) < 5:
            continue
        if _is_header(norm):
            in_table = True
            continue
        if _SUMMARY_START.match(norm):
            in_table = False
            continue

        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 4:
                item = _make_item(*parts)
                if item:
                    items.append(item)
                    continue
            if len(parts) == 2 and allow_quantity_only:
                item = _make_qty_item(*parts)
                if item:
                    items.append(item)
                    continue
            if len(parts) > 4 and in_table:
                unparsed.append(norm)       # extra columns: we will not guess which is which
            continue

        if in_table:
            m = _WS_ROW.match(line)
            item = _make_item(m["name"], m["qty"], m["price"], m["total"]) if m else None
            if item is None and allow_quantity_only:
                q = _WS_QTY_ONLY.match(line)
                item = _make_qty_item(q["name"], q["qty"]) if q else None
            if item:
                items.append(item)
            elif re.search(r"\d", norm):
                unparsed.append(norm)

    for row in unparsed[:5]:
        warnings.append(f"Could not read this table row, please add it by hand: “{row}”")
    if len(unparsed) > 5:
        warnings.append(f"{len(unparsed) - 5} more table rows could not be read.")
    return items, warnings


# -------------------------------------------------------------- main entry
def extract_fields(text: str, document_type: str) -> tuple[dict, list[str]]:
    """Return (data, warnings). Never raises on odd text; unknown values stay None."""
    data = empty_document(document_type)
    if not text or not text.strip():
        return data, ["No text available for field extraction."]
    warnings: list[str] = []

    data["document_id"] = _find_document_id(text, document_type)
    data["po_reference"] = _find_po_reference(text, document_type)
    data["delivery_date"] = _find_delivery_date(text)
    data["vendor"] = _label_value(VENDOR_PATTERNS, text)
    data["document_date"], ambiguous = _find_date(text)
    if ambiguous:
        warnings.append("The date could be day/month or month/day. It was read as day/month; please check.")
    data["currency"], conflict = _find_currency(text)
    if conflict:
        warnings.append("More than one currency appears in the document; currency was left empty.")

    data["items"], item_warnings = extract_items(
        text, allow_quantity_only=(document_type == "Delivery Receipt / GRN"))
    warnings.extend(item_warnings)
    summary, summary_warnings = _extract_summary(text)
    data.update(summary)
    warnings.extend(summary_warnings)
    return data, warnings
