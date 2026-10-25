"""Optional Gemini reader: sends a PDF/image straight to Gemini and gets back structured fields.

Why: PaddleOCR is heavy (and often not installable on Streamlit Cloud), and rules only
work on clean layouts. Gemini can read photos (e.g. WhatsApp images), tell what kind of
document it is, and return line items. Output is normalised through the same parsers the
rule-based path uses, so everything downstream (validation, DB, matching) is unchanged.

Setup: add GEMINI_API_KEY (and optionally GEMINI_MODEL) to Streamlit secrets or the environment.
"""
import json
import os
import re
import time

from .extract_fields import (ITEM_KEYS, _amount_str, empty_document, parse_date)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODELS = ("gemini-2.5-flash-lite",)   # tried when the main model stays overloaded
RETRY_DELAYS = (2, 5, 10)                              # seconds to wait before retry 1, 2, 3
_TRANSIENT = ("503", "unavailable", "overloaded", "429", "resource_exhausted", "500", "504",
              "deadline", "timed out", "timeout", "connection")
DOC_TYPES = ("Invoice", "Purchase Order", "Delivery Receipt / GRN")
MIME = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

PROMPT = """You read procurement documents. The file is ONE document: a purchase order, a goods
receipt note / delivery receipt (GRN), or a supplier invoice. Return ONLY a JSON object:

{
  "document_type": "Invoice" | "Purchase Order" | "Delivery Receipt / GRN",
  "document_id": "the document's own number (invoice no, PO no or GRN no)",
  "po_reference": "the PO number this document refers to (null on a PO itself)",
  "vendor": "supplier name",
  "document_date": "YYYY-MM-DD",
  "delivery_date": "YYYY-MM-DD or null",
  "currency": "INR / USD / ... or null",
  "items": [{"name": "...", "quantity": "number", "unit_price": "number or null", "line_total": "number or null"}],
  "subtotal": "number or null", "tax": "number or null", "total": "number or null"
}

Rules: copy values exactly as printed; never invent or calculate a value that is not on the page
(use null). Numbers are plain strings without currency symbols or thousands separators.
Dates are ISO (YYYY-MM-DD); read numeric dates day-first. A GRN usually has quantities only.
"""


class GeminiUnavailableError(RuntimeError):
    """No API key, or the google-genai package is missing."""


class GeminiError(RuntimeError):
    """Gemini was called but failed or returned unusable output."""


def get_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:                                   # Streamlit Cloud secrets
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def get_model() -> str:
    model = os.environ.get("GEMINI_MODEL")
    if not model:
        try:
            import streamlit as st
            model = st.secrets.get("GEMINI_MODEL")
        except Exception:
            model = None
    return model or DEFAULT_MODEL


def get_fallback_models() -> list[str]:
    raw = os.environ.get("GEMINI_FALLBACK_MODELS")
    if raw is None:
        try:
            import streamlit as st
            raw = st.secrets.get("GEMINI_FALLBACK_MODELS")
        except Exception:
            raw = None
    models = DEFAULT_FALLBACK_MODELS if raw is None else [m.strip() for m in str(raw).split(",") if m.strip()]
    return [m for m in models if m != get_model()]


def is_transient(exc: Exception) -> bool:
    """True for temporary problems (overload, rate limit, timeout) that are worth retrying."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT)


def call_with_retry(call, models: list[str], delays=RETRY_DELAYS, sleep=time.sleep):
    """call(model) -> text. Retries temporary errors on each model, then moves to the next model.

    Permanent errors (bad key, bad request) stop immediately. Raises GeminiError at the end.
    """
    last = None
    for model in models:
        for attempt in range(len(delays) + 1):
            try:
                return call(model)
            except Exception as exc:
                last = exc
                if not is_transient(exc):
                    raise GeminiError(f"Gemini request failed: {exc}") from exc
                if attempt < len(delays):
                    sleep(delays[attempt])
        # this model stayed busy; fall through to the next one
    raise GeminiError(f"Gemini is busy right now (tried {', '.join(models)}). Last error: {last}")


def is_gemini_available() -> bool:
    if not get_api_key():
        return False
    try:
        import google.genai  # noqa: F401
    except Exception:
        return False
    return True


def parse_json_reply(text: str) -> dict:
    """Pull one JSON object out of a model reply (tolerates ``` fences and chatter)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            raise GeminiError("Gemini did not return JSON.")
        try:
            value = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise GeminiError(f"Gemini returned malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GeminiError("Gemini returned JSON, but not an object.")
    return value


def _clean_date(value) -> str | None:
    if not value:
        return None
    iso, _ = parse_date(str(value))
    return iso


def normalize(raw: dict, fallback_type: str | None = None) -> dict:
    """Map Gemini's JSON onto the app's schema. Anything unparseable becomes None."""
    doc_type = raw.get("document_type")
    if doc_type not in DOC_TYPES:
        doc_type = fallback_type if fallback_type in DOC_TYPES else None
    data = empty_document(doc_type)

    def text(key):
        v = raw.get(key)
        v = str(v).strip() if v not in (None, "") else None
        return v or None

    data["document_id"] = text("document_id")
    data["po_reference"] = text("po_reference")
    data["vendor"] = text("vendor")
    data["currency"] = (text("currency") or "").upper() or None
    data["document_date"] = _clean_date(raw.get("document_date"))
    data["delivery_date"] = _clean_date(raw.get("delivery_date"))
    for key in ("subtotal", "tax", "total"):
        data[key] = _amount_str(raw.get(key))
    items = []
    for item in raw.get("items") or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        items.append({"name": str(item["name"]).strip(),
                      "quantity": _amount_str(item.get("quantity")),
                      "unit_price": _amount_str(item.get("unit_price")),
                      "line_total": _amount_str(item.get("line_total"))})
    data["items"] = [{k: i.get(k) for k in ITEM_KEYS} for i in items]
    return data


TEXT_PROMPT = PROMPT + """
The input is text produced by OCR from the document, not the document itself. Table columns are
separated by 3 or more spaces. The text may contain OCR mistakes: fix a character only when the
surrounding numbers make it certain (for example a quantity x price = line total check), otherwise
keep it as written. Never add values that are not in the text.

OCR TEXT:
"""


def _client():
    key = get_api_key()
    if not key:
        raise GeminiUnavailableError("No Gemini API key. Add GEMINI_API_KEY to the Streamlit secrets.")
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise GeminiUnavailableError("The google-genai package is not installed.") from exc
    return genai.Client(api_key=key), types


def _generate(contents_for, document_type: str | None) -> dict:
    client, types = _client()

    def call(model: str) -> str:
        response = client.models.generate_content(
            model=model, contents=contents_for(types),
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
        return getattr(response, "text", "") or ""

    text = call_with_retry(call, [get_model(), *get_fallback_models()])
    return normalize(parse_json_reply(text), document_type)


def _hint(document_type: str | None) -> str:
    return f"\nThe user says this is a: {document_type}." if document_type in DOC_TYPES else ""


def gemini_format_text(text: str, document_type: str | None = None) -> dict:
    """OCR/PDF text -> schema-shaped data. Raises GeminiUnavailableError / GeminiError."""
    return _generate(lambda types: [TEXT_PROMPT + text + "\n" + _hint(document_type)], document_type)


def gemini_extract(file_bytes: bytes, suffix: str, document_type: str | None = None) -> dict:
    """Last resort when no text could be read: Gemini reads the image/PDF itself."""
    return _generate(lambda types: [types.Part.from_bytes(data=file_bytes, mime_type=MIME[suffix]),
                                    PROMPT + _hint(document_type)], document_type)
