"""File bytes -> text -> structured data. Never raises for bad input."""
import hashlib
import re
from pathlib import Path

import pymupdf  # PyMuPDF

from .extract_fields import extract_fields
from .gemini_reader import (GeminiError, GeminiUnavailableError, gemini_extract,
                            gemini_format_text, get_model)
from .ocr_reader import (OCRError, OCRUnavailableError, ocr_image_bytes,
                         ocr_pdf_bytes)
from .validation import validate_document

MAX_FILE_BYTES = 10 * 1024 * 1024        # 10 MB
MAX_PDF_PAGES = 10
MIN_USABLE_CHARS = 30                    # fewer letters/digits than this = scanned PDF
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
AUTO = "Auto-detect"
_TYPE_WORDS = (("Purchase Order", r"purchase\s*order|\bP\.?O\.?\s*(?:no|number|#)"),
               ("Delivery Receipt / GRN", r"goods\s*receipt|\bGRN\b|delivery\s*(?:receipt|note|challan)"),
               ("Invoice", r"\binvoice\b|\bbill\s*(?:no|number)"))
OCR_HINT = "OCR is optional and not installed. Run: pip install -r requirements-ocr.txt"


def pdf_page_text(page) -> str:
    """Page text with table columns kept on one line.

    Words are grouped by vertical position; a wide horizontal gap becomes 3
    spaces so the extractor can see the column boundaries.
    """
    words = page.get_text("words")
    if not words:
        return ""
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines, current, current_y = [], [], None
    for w in words:
        y = (w[1] + w[3]) / 2
        height = max(w[3] - w[1], 1.0)
        if current and abs(y - current_y) > height * 0.5:
            lines.append(current)
            current = []
        if not current:
            current_y = y
        current.append(w)
    if current:
        lines.append(current)
    out = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        text = line[0][4]
        for prev, cur in zip(line, line[1:]):
            height = max(cur[3] - cur[1], 1.0)
            text += ("   " if cur[0] - prev[2] > height * 1.2 else " ") + cur[4]
        out.append(text)
    return "\n".join(out)


def guess_document_type(text: str) -> str | None:
    """Best-effort type from the title area (first lines), used only when type is Auto-detect."""
    head = "\n".join((text or "").splitlines()[:8])
    for name, pattern in _TYPE_WORDS:
        if re.search(pattern, head, flags=re.IGNORECASE):
            return name
    return None


def _usable(text: str) -> bool:
    return len(re.sub(r"\W+", "", text)) >= MIN_USABLE_CHARS


def read_pdf_text(file_bytes: bytes) -> str:
    """Embedded text of a digital PDF. Raises ValueError with a friendly message."""
    try:
        pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception:
        raise ValueError("This PDF is corrupt or cannot be opened.")
    with pdf:
        if pdf.needs_pass or pdf.is_encrypted:
            raise ValueError("This PDF is password-protected. Remove the password and upload again.")
        if pdf.page_count == 0:
            raise ValueError("This PDF has no pages.")
        pages = [pdf_page_text(p) or p.get_text("text") for p in pdf if p.number < MAX_PDF_PAGES]
    return "\n".join(pages)


def _looks_like(file_bytes: bytes, suffix: str) -> bool:
    """Cheap check that the content matches the extension (magic bytes)."""
    if suffix in PDF_SUFFIXES:
        return file_bytes[:5] == b"%PDF-"
    if suffix == ".png":
        return file_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    return file_bytes[:3] == b"\xff\xd8\xff"          # JPEG


def process_document(file_bytes: bytes, filename: str, document_type: str, use_gemini: bool = False) -> dict:
    """Return {filename, text_source, raw_text, data, warnings}. Errors become warnings.

    Flow: 1) read text (PyMuPDF for digital PDFs, PaddleOCR for photos/scans)
          2) with use_gemini, Gemini turns that text into the app's JSON format
          3) if Gemini fails, the rule-based extractor works on the same text.
    If no text could be read at all (e.g. OCR not installed), Gemini reads the image directly.
    document_type may be AUTO ("Auto-detect").
    """
    from .extract_fields import empty_document
    suffix = Path(filename or "").suffix.lower()
    warnings: list[str] = []
    raw_text, text_source = "", "None"

    problem = None
    if not file_bytes:
        problem = "The file is empty."
    elif len(file_bytes) > MAX_FILE_BYTES:
        problem = f"The file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB."
    elif suffix not in PDF_SUFFIXES | IMAGE_SUFFIXES:
        problem = "Unsupported file type. Use PDF, PNG, JPG or JPEG."
    elif not _looks_like(file_bytes, suffix):
        problem = "The file content does not match its extension, so it was not processed."

    # 1) text
    if problem:
        warnings.append(problem)
    elif suffix in PDF_SUFFIXES:
        try:
            raw_text, text_source = read_pdf_text(file_bytes), "PyMuPDF (digital PDF)"
        except ValueError as exc:
            warnings.append(str(exc))
        if not warnings and not _usable(raw_text):
            warnings.append("The PDF has little or no embedded text; trying OCR.")
            raw_text, text_source = _try_ocr(lambda: ocr_pdf_bytes(file_bytes, MAX_PDF_PAGES),
                                             warnings, "PaddleOCR (scanned PDF)")
    else:
        raw_text, text_source = _try_ocr(lambda: ocr_image_bytes(file_bytes, suffix),
                                         warnings, "PaddleOCR (image)")

    # 2) Gemini formats the text (or reads the image itself if there is no text)
    gemini_data = None
    if not problem and use_gemini:
        try:
            wanted = None if document_type == AUTO else document_type
            if _usable(raw_text):
                gemini_data = gemini_format_text(raw_text, wanted)
                text_source = f"{text_source} + Gemini ({get_model()})"
            else:
                gemini_data = gemini_extract(file_bytes, suffix, wanted)
                text_source = f"Gemini ({get_model()}), image read directly"
        except (GeminiUnavailableError, GeminiError) as exc:
            warnings.append(f"Gemini could not format this file ({exc}). The built-in rules were used instead.")

    # 3) result
    field_warnings: list[str] = []
    if gemini_data is not None:
        data, has_content = gemini_data, True
        if not raw_text.strip():
            raw_text = "(Read directly from the image by Gemini; there is no separate text layer.)"
        found = data.get("document_type")
        if document_type != AUTO and found and found != document_type:
            warnings.append(f"You chose {document_type}, but Gemini thinks this is a {found}. Please check.")
        effective = document_type if document_type != AUTO else (found or "Invoice")
        if document_type == AUTO and not found:
            warnings.append("The document type could not be detected; Invoice was assumed. Please check.")
    else:
        effective = document_type
        if document_type == AUTO:
            guessed = guess_document_type(raw_text)
            effective = guessed or "Invoice"
            if guessed is None and raw_text.strip():
                warnings.append("The document type could not be detected; Invoice was assumed. Please check.")
        has_content = bool(raw_text.strip()) and not problem
        if not has_content:
            data = empty_document(effective)
            if not problem and not any("No text" in w for w in warnings):
                warnings.append("No usable text was found in this file.")
        else:
            data, field_warnings = extract_fields(raw_text, effective)
    data["source_filename"] = filename
    data["document_type"] = effective
    warnings.extend(field_warnings)
    if has_content:
        warnings.extend(validate_document(data))
    return {"filename": filename, "text_source": text_source, "raw_text": raw_text,
            "data": data, "warnings": warnings,
            "file_sha256": hashlib.sha256(file_bytes).hexdigest() if file_bytes else None}


def _try_ocr(run, warnings: list[str], label: str) -> tuple[str, str]:
    try:
        return run(), label
    except OCRUnavailableError:
        warnings.append(OCR_HINT)
    except OCRError as exc:
        warnings.append(str(exc))
    except Exception as exc:                      # OCR must never crash the app
        warnings.append(f"OCR failed: {exc}")
    return "", "None"
