import json

import pytest

from src import gemini_reader, pipeline
from src.gemini_reader import GeminiError, normalize, parse_json_reply

REPLY = {
    "document_type": "Invoice", "document_id": "INV-7001", "po_reference": "PO-5001",
    "vendor": "Test Traders", "document_date": "20/10/2026", "delivery_date": None, "currency": "inr",
    "items": [{"name": "Steel Bolt", "quantity": "100", "unit_price": "₹2.50", "line_total": "250.00"},
              {"name": "", "quantity": "1"}],
    "subtotal": "250.00", "tax": "45.00", "total": "295.00",
}


def test_parse_json_reply_handles_fences_and_chatter():
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_reply('Here you go: {"a": 1} hope it helps') == {"a": 1}
    with pytest.raises(GeminiError):
        parse_json_reply("no json here")
    with pytest.raises(GeminiError):
        parse_json_reply("[1, 2]")


def test_normalize_maps_to_schema():
    data = normalize(REPLY)
    assert data["document_type"] == "Invoice" and data["currency"] == "INR"
    assert data["document_date"] == "2026-10-20"
    assert data["items"] == [{"name": "Steel Bolt", "quantity": "100", "unit_price": "2.50", "line_total": "250.00"}]
    assert data["total"] == "295.00" and json.loads(json.dumps(data)) == data


def test_normalize_bad_values_become_none():
    data = normalize({"document_type": "Receipt??", "total": "1.234,56", "document_date": "someday"}, "Purchase Order")
    assert data["document_type"] == "Purchase Order" and data["total"] is None and data["document_date"] is None


def _fake(reply):
    def run(file_bytes, suffix, document_type=None):
        return normalize(reply, document_type)
    return run


PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20


def test_image_without_ocr_text_gemini_reads_image(monkeypatch):
    monkeypatch.setattr(pipeline, "ocr_image_bytes", lambda b, s: "")
    monkeypatch.setattr(pipeline, "gemini_extract", _fake(REPLY))
    result = pipeline.process_document(PNG, "wa.png", pipeline.AUTO, use_gemini=True)
    assert result["text_source"].startswith("Gemini")
    assert result["data"]["document_type"] == "Invoice" and result["data"]["document_id"] == "INV-7001"
    assert result["data"]["source_filename"] == "wa.png"


def test_user_choice_wins_but_mismatch_is_flagged(monkeypatch):
    monkeypatch.setattr(pipeline, "ocr_image_bytes", lambda b, s: "")
    monkeypatch.setattr(pipeline, "gemini_extract", _fake(REPLY))
    result = pipeline.process_document(PNG, "wa.png", "Purchase Order", use_gemini=True)
    assert result["data"]["document_type"] == "Purchase Order"
    assert any("Gemini thinks" in w for w in result["warnings"])


OCR_TEXT = "TAX INVOICE\nInvoice No: INV-9\nVendor: Acme Ltd\nInvoice Date: 2026-10-20\nGrand Total: 118.00"


def test_ocr_text_goes_to_gemini_then_result(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "ocr_image_bytes", lambda b, s: OCR_TEXT)
    def fmt(text, document_type=None):
        seen["text"] = text
        return normalize(REPLY, document_type)
    monkeypatch.setattr(pipeline, "gemini_format_text", fmt)
    result = pipeline.process_document(PNG, "wa.png", pipeline.AUTO, use_gemini=True)
    assert seen["text"] == OCR_TEXT
    assert result["text_source"].startswith("PaddleOCR (image) + Gemini")
    assert result["data"]["document_id"] == "INV-7001"


def test_gemini_failure_uses_rules_on_ocr_text(monkeypatch):
    monkeypatch.setattr(pipeline, "ocr_image_bytes", lambda b, s: OCR_TEXT)
    def boom(*a, **k):
        raise GeminiError("busy")
    monkeypatch.setattr(pipeline, "gemini_format_text", boom)
    result = pipeline.process_document(PNG, "wa.png", pipeline.AUTO, use_gemini=True)
    assert any("built-in rules" in w for w in result["warnings"])
    assert result["data"]["document_id"] == "INV-9" and result["data"]["document_type"] == "Invoice"


def test_gemini_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise GeminiError("quota")
    monkeypatch.setattr(pipeline, "ocr_image_bytes", lambda b, s: "")
    monkeypatch.setattr(pipeline, "gemini_extract", boom)
    result = pipeline.process_document(PNG, "wa.png", "Invoice", use_gemini=True)
    assert any("Gemini could not format" in w for w in result["warnings"])
    assert result["data"]["document_type"] == "Invoice"


def test_auto_detect_without_gemini_uses_title():
    assert pipeline.guess_document_type("PURCHASE ORDER\nPO No: PO-1") == "Purchase Order"
    assert pipeline.guess_document_type("Goods Receipt Note\nGRN No: G-1") == "Delivery Receipt / GRN"
    assert pipeline.guess_document_type("TAX INVOICE\nInvoice No: 5") == "Invoice"
    assert pipeline.guess_document_type("hello") is None


def test_is_gemini_available_needs_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(gemini_reader, "get_api_key", lambda: None)
    assert gemini_reader.is_gemini_available() is False


def test_retry_then_succeed():
    calls = []
    def call(model):
        calls.append(model)
        if len(calls) < 3:
            raise RuntimeError("503 UNAVAILABLE high demand")
        return "ok"
    waits = []
    assert gemini_reader.call_with_retry(call, ["main"], delays=(1, 2, 3), sleep=waits.append) == "ok"
    assert calls == ["main"] * 3 and waits == [1, 2]


def test_switches_to_backup_model_when_main_stays_busy():
    def call(model):
        if model == "main":
            raise RuntimeError("503 UNAVAILABLE")
        return "from backup"
    assert gemini_reader.call_with_retry(call, ["main", "backup"], delays=(0, 0), sleep=lambda s: None) == "from backup"


def test_permanent_error_does_not_retry():
    calls = []
    def call(model):
        calls.append(model)
        raise RuntimeError("400 INVALID_ARGUMENT API key not valid")
    with pytest.raises(GeminiError):
        gemini_reader.call_with_retry(call, ["main", "backup"], delays=(0, 0), sleep=lambda s: None)
    assert calls == ["main"]


def test_all_models_busy_raises():
    with pytest.raises(GeminiError):
        gemini_reader.call_with_retry(lambda m: (_ for _ in ()).throw(RuntimeError("503")), ["a", "b"],
                                      delays=(0,), sleep=lambda s: None)
