"""Document extraction page: upload several documents at once, review them, save them, match them."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import db
from src.analysis import match_all_invoices
from src.gemini_reader import get_model, is_gemini_available
from src.ocr_reader import is_ocr_available
from src.pipeline import AUTO, MAX_FILE_BYTES, process_document
from src.validation import validate_structure
from views.common import field_card, header, money, status_box

DOC_TYPES = ["Invoice", "Purchase Order", "Delivery Receipt / GRN"]
SAVE_ORDER = {"Purchase Order": 0, "Delivery Receipt / GRN": 1, "Invoice": 2}   # PO first, invoice last


def _batch() -> dict:
    """The documents read so far. Lives in session_state, so adding files never drops earlier ones."""
    return st.session_state.setdefault("batch", {})


def _current_json(sha: str, result: dict):
    """(parsed dict or None, error text or None) for the edited JSON of one document."""
    text = st.session_state.get(f"json_{sha}")
    if text is None:
        text = json.dumps(result["data"], indent=2, ensure_ascii=False)
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _extract_new(files, doc_type: str, use_gemini: bool) -> None:
    batch = _batch()
    todo = []
    for f in files:
        raw = f.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        if sha not in batch:
            todo.append((sha, f.name, raw))
    if not todo:
        st.info("All uploaded files have already been read.")
        return
    bar = st.progress(0.0, text="Reading documents...")
    for i, (sha, name, raw) in enumerate(todo, 1):
        bar.progress((i - 1) / len(todo), text=f"Reading {name} ({i}/{len(todo)})")
        try:
            result = process_document(raw, name, doc_type, use_gemini=use_gemini)
        except Exception as exc:                     # last safety net: one bad file must not stop the rest
            result = {"filename": name, "text_source": "None", "raw_text": "", "file_sha256": sha,
                      "data": {**{k: None for k in ("document_id", "po_reference", "vendor", "document_date",
                                                    "delivery_date", "currency", "subtotal", "tax", "total")},
                               "document_type": None if doc_type == AUTO else doc_type,
                               "items": [], "source_filename": name},
                      "warnings": [f"Could not process this file: {exc}"]}
        batch[sha] = {"result": result, "saved": None}
    bar.empty()


def _save_all() -> None:
    rows = []
    batch = _batch()
    pending = sorted(((sha, e) for sha, e in batch.items() if not e["saved"]),
                     key=lambda it: SAVE_ORDER.get((it[1]["result"]["data"].get("document_type")), 3))
    with db.get_conn() as conn:
        for sha, entry in pending:
            result = entry["result"]
            parsed, err = _current_json(sha, result)
            if parsed is None:
                rows.append((result["filename"], f"Not saved: JSON is not valid ({err})"))
                continue
            problems = validate_structure(parsed)
            if problems:
                rows.append((result["filename"], f"Not saved: {problems[0]}"))
                continue
            try:
                kind, key = db.save_document(conn, parsed, result.get("file_sha256"))
                entry["saved"] = (kind, key, parsed.get("document_id"))
                rows.append((result["filename"], f"Saved {kind} {parsed.get('document_id')}"))
            except ValueError as exc:
                rows.append((result["filename"], f"Not saved: {exc}"))
    st.session_state["save_report"] = rows


def _document_view(data: dict) -> None:
    a, b, c = st.columns(3)
    a.markdown(field_card("Document ID", data.get("document_id")), unsafe_allow_html=True)
    b.markdown(field_card("Vendor", data.get("vendor")), unsafe_allow_html=True)
    c.markdown(field_card("Date", data.get("document_date")), unsafe_allow_html=True)
    d, e = st.columns(2)
    d.markdown(field_card("PO reference", data.get("po_reference")), unsafe_allow_html=True)
    e.markdown(field_card("Delivery date", data.get("delivery_date")), unsafe_allow_html=True)
    if data.get("items"):
        st.dataframe(pd.DataFrame(data["items"]), width="stretch", hide_index=True)
    else:
        st.info("No line items detected.")
    cur = data.get("currency")
    st.markdown(
        f'<div class="totals"><div><span>Subtotal</span><span>{money(cur, data.get("subtotal"))}</span></div>'
        f'<div><span>Tax</span><span>{money(cur, data.get("tax"))}</span></div>'
        f'<div class="grand"><span>Total</span><span>{money(cur, data.get("total"))}</span></div></div>',
        unsafe_allow_html=True)


def _review_one(sha: str, entry: dict) -> None:
    result = entry["result"]
    data = result["data"]
    saved = entry["saved"]
    title = f'{"✅" if saved else "📄"} {result["filename"]} · {data.get("document_type") or "?"} · {data.get("document_id") or "no ID"}'
    with st.expander(title, expanded=not saved):
        st.caption(f'Read with: {result["text_source"]}')
        if result["warnings"]:
            for w in result["warnings"]:
                st.warning(w)
        tab_doc, tab_json, tab_raw = st.tabs(["Document view", "Edit", "Raw text"])
        with tab_doc:
            parsed, _ = _current_json(sha, result)
            _document_view(parsed if isinstance(parsed, dict) else data)
        with tab_json:
            if f"json_{sha}" not in st.session_state:
                st.session_state[f"json_{sha}"] = json.dumps(data, indent=2, ensure_ascii=False)
            st.text_area("Correct any extraction errors here", key=f"json_{sha}", height=320,
                         disabled=bool(saved))
            parsed, err = _current_json(sha, result)
            if err:
                st.error(f"JSON is not valid yet: {err}")
            else:
                for problem in validate_structure(parsed):
                    st.warning(problem)
                st.download_button("Download JSON", data=json.dumps(parsed, indent=2, ensure_ascii=False),
                                   file_name=f'{Path(result["filename"]).stem}_extracted.json',
                                   mime="application/json", key=f"dl_{sha}")
        with tab_raw:
            st.text(result["raw_text"] or "(No text could be extracted.)")
        if st.button("Remove from this batch", key=f"rm_{sha}"):
            _batch().pop(sha, None)
            st.session_state.pop(f"json_{sha}", None)
            st.rerun()


def _batch_match() -> None:
    """Three-way match for the invoices saved from this batch."""
    saved_ids = {e["saved"][1] for e in _batch().values() if e["saved"] and e["saved"][0] == "Invoice"}
    if not saved_ids:
        return
    st.subheader("Three-way match for this batch")
    with db.get_conn() as conn:
        results = [m for m in match_all_invoices(conn) if str(m["invoice_id"]) in saved_ids]
    for m in results:
        st.markdown(f"**Invoice {m['invoice']}** → PO {m['po']}")
        status_box(m["result"]["status"])
        if m["result"]["discrepancies"]:
            st.dataframe(pd.DataFrame(m["result"]["discrepancies"])[["severity", "message", "expected", "actual"]],
                         width="stretch", hide_index=True)
    st.caption("Open the Three-way match page for the item-by-item comparison.")


def render() -> None:
    header("Extract documents", "Upload the PO, GRN and invoice together. They are read as one batch, reviewed, saved and matched.")
    gemini_ok = is_gemini_available()
    c1, c2 = st.columns([2, 3])
    with c1:
        use_gemini = st.toggle("Read with Gemini", value=gemini_ok, disabled=not gemini_ok,
                               help="OCR/PDF text is sent to Gemini, which formats it into the app's fields and detects the document type.")
    with c2:
        if gemini_ok:
            st.caption(f"Gemini is ready ({get_model()}). If it fails for a file, the built-in reader is used.")
        else:
            st.caption("Gemini is off: add GEMINI_API_KEY to the app secrets (and install google-genai). "
                       + ("OCR is installed, so photos still work." if is_ocr_available()
                          else "Without OCR, only digital PDFs can be read."))

    doc_type = st.radio("Document type for new files", [AUTO] + DOC_TYPES, horizontal=True,
                        help="Auto-detect is best when you upload a mix of PO, GRN and invoice.")
    files = st.file_uploader(f"Upload PDF, PNG, JPG or JPEG (max {MAX_FILE_BYTES // (1024 * 1024)} MB each)",
                             type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, key="uploader")
    batch = _batch()
    new_count = sum(1 for f in files or [] if hashlib.sha256(f.getvalue()).hexdigest() not in batch)
    if files and st.button(f"Extract {new_count} new file(s)" if new_count else "Nothing new to extract",
                           type="primary", disabled=not new_count):
        _extract_new(files, doc_type, use_gemini and gemini_ok)
        st.rerun()

    if not batch:
        return
    st.divider()
    st.subheader(f"This batch ({len(batch)} document{'s' if len(batch) != 1 else ''})")
    st.caption("Earlier documents stay here when you add more files. Extracted values may contain mistakes, so review them.")
    st.dataframe(pd.DataFrame([{
        "File": e["result"]["filename"],
        "Type": e["result"]["data"].get("document_type"),
        "ID": e["result"]["data"].get("document_id"),
        "PO ref": e["result"]["data"].get("po_reference"),
        "Total": e["result"]["data"].get("total"),
        "Notes": len(e["result"]["warnings"]),
        "Saved": "✅" if e["saved"] else "",
    } for e in batch.values()]), width="stretch", hide_index=True)

    for sha, entry in list(batch.items()):
        _review_one(sha, entry)

    left, right = st.columns(2)
    unsaved = sum(1 for e in batch.values() if not e["saved"])
    if left.button(f"Save {unsaved} unsaved document(s) to database", type="primary", disabled=not unsaved):
        _save_all()
        st.rerun()
    if right.button("Clear batch"):
        for sha in list(batch):
            st.session_state.pop(f"json_{sha}", None)
        batch.clear()
        st.session_state.pop("save_report", None)
        st.rerun()

    for name, message in st.session_state.get("save_report", []):
        (st.success if message.startswith("Saved") else st.error)(f"{name}: {message}")
    _batch_match()
