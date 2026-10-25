"""Optional OCR using PaddleOCR 3.x (pinned to >=3.0,<4 in requirements-ocr.txt).

PaddleOCR is imported lazily, so the app runs for digital PDFs without it.
Verified against paddleocr 3.7.0: PaddleOCR(use_textline_orientation=...,
use_doc_orientation_classify=..., use_doc_unwarping=..., lang=...) and
engine.predict(path) -> list of results holding 'rec_texts' and 'rec_boxes'/'rec_polys'.
"""
import os
import ctypes.util
import statistics
import tempfile

PDF_DPI = 200
_engine = None


class OCRUnavailableError(RuntimeError):
    """PaddleOCR is not installed."""


class OCRError(RuntimeError):
    """PaddleOCR is installed but failed on this file."""


_paddle_cls = None
_paddle_error: str | None = None


def _load_paddle():
    """Import PaddleOCR once. A failed import is remembered, never retried
    (retrying causes the misleading 'PDX has already been initialized' error)."""
    global _paddle_cls, _paddle_error
    if _paddle_cls is not None:
        return _paddle_cls
    if _paddle_error is not None:
        raise OCRError(_paddle_error)
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        _paddle_error = f"PaddleOCR import failed: {type(exc).__name__}: {exc}"
        print("OCR DEBUG -", _paddle_error)
        raise OCRError(_paddle_error) from exc
    _paddle_cls = PaddleOCR
    return _paddle_cls


def is_ocr_available() -> bool:
    try:
        _load_paddle()
        return True
    except OCRError:
        return False


def _get_engine():
    global _engine
    if _engine is None:
        PaddleOCR = _load_paddle()
        try:
            _engine = PaddleOCR(
                lang="en",
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except Exception as exc:
            raise OCRError(f"Could not start PaddleOCR: {exc}") from exc
    return _engine

def boxes_to_text(texts: list[str], boxes: list) -> str:
    """Rebuild text lines from OCR boxes so table columns stay on one line.

    Each box is (x0, y0, x1, y1). Boxes whose vertical centres are close belong
    to the same line; a large horizontal gap becomes 3 spaces (a column gap).
    Pure function, so it is easy to unit-test without PaddleOCR.
    """
    entries = []
    for text, box in zip(texts, boxes):
        text = str(text).strip()
        if text:
            x0, y0, x1, y1 = (float(v) for v in box)
            entries.append((text, x0, y0, x1, y1))
    if not entries:
        return ""
    height = statistics.median(e[4] - e[2] for e in entries) or 10.0
    entries.sort(key=lambda e: ((e[2] + e[4]) / 2, e[1]))
    lines: list[list[tuple]] = []
    for entry in entries:
        centre = (entry[2] + entry[4]) / 2
        if lines and abs(centre - _line_centre(lines[-1])) <= height * 0.6:
            lines[-1].append(entry)
        else:
            lines.append([entry])
    out = []
    for line in lines:
        line.sort(key=lambda e: e[1])
        pieces = [line[0][0]]
        for prev, cur in zip(line, line[1:]):
            gap = cur[1] - prev[3]
            pieces.append(("   " if gap > height * 1.5 else " ") + cur[0])
        out.append("".join(pieces))
    return "\n".join(out)


def _line_centre(line) -> float:
    return sum((e[2] + e[4]) / 2 for e in line) / len(line)


def _poly_to_box(poly) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _result_to_text(results) -> str:
    """Turn PaddleOCR 3.x predict() output into text (one string per page)."""
    pages = []
    for res in results or []:
        texts = list(res["rec_texts"]) if "rec_texts" in res else []
        boxes = None
        if "rec_boxes" in res and len(res["rec_boxes"]):
            boxes = [tuple(b) for b in res["rec_boxes"]]
        elif "rec_polys" in res and len(res["rec_polys"]):
            boxes = [_poly_to_box(p) for p in res["rec_polys"]]
        if boxes is not None and len(boxes) == len(texts):
            pages.append(boxes_to_text(texts, boxes))
        else:
            pages.append("\n".join(str(t) for t in texts))
    return "\n".join(p for p in pages if p)


def ocr_image_file(path: str) -> str:
    engine = _get_engine()
    try:
        return _result_to_text(engine.predict(path))
    except (OCRUnavailableError, OCRError):
        raise
    except Exception as exc:
        raise OCRError(f"OCR failed: {exc}") from exc


def ocr_image_bytes(image_bytes: bytes, suffix: str = ".png") -> str:
    """OCR PNG/JPG bytes. The file is only ever read as an image, never executed."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            path = tmp.name
        return ocr_image_file(path)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = 10) -> str:
    """OCR a scanned PDF: draw each page as an image with PyMuPDF, then OCR it."""
    import pymupdf
    texts = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for index, page in enumerate(pdf):
            if index >= max_pages:
                break
            png = page.get_pixmap(dpi=PDF_DPI).tobytes("png")
            texts.append(ocr_image_bytes(png, ".png"))
    return "\n".join(t for t in texts if t)
