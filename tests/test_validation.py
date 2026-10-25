from src.validation import validate_amounts, validate_document, validate_structure


def doc(**kw):
    base = {"document_type": "Invoice", "document_id": "A1", "po_reference": "PO-1", "vendor": "V",
            "delivery_date": None, "document_date": "2026-10-20",
            "currency": "INR", "items": [], "subtotal": None, "tax": None, "total": None,
            "source_filename": "x.pdf"}
    base.update(kw)
    return base


def item(q="10", p="500.00", t="5000.00", name="Mouse"):
    return {"name": name, "quantity": q, "unit_price": p, "line_total": t}


def test_matching_line_has_no_warning():
    assert validate_amounts(doc(items=[item()])) == []


def test_line_total_mismatch_warns():
    warnings = validate_amounts(doc(items=[item(t="4000.00")]))
    assert len(warnings) == 1 and "Mouse" in warnings[0]


def test_rounding_difference_allowed():
    assert validate_amounts(doc(items=[item("3", "3.333", "10.00")])) == []   # 9.999 vs 10.00


def test_items_vs_subtotal():
    warnings = validate_amounts(doc(items=[item()], subtotal="4000.00"))
    assert any("subtotal" in w for w in warnings)
    assert validate_amounts(doc(items=[item()], subtotal="5000.00")) == []


def test_subtotal_plus_tax_vs_total():
    assert validate_amounts(doc(subtotal="100.00", tax="18.00", total="118.00")) == []
    assert validate_amounts(doc(subtotal="100.00", tax="18.00", total="120.00"))


def test_structure_errors():
    assert validate_structure("not a dict")
    assert any("items" in p for p in validate_structure({"items": "x"}))
    assert any("total" in p for p in validate_structure(doc(total="abc")))
    assert any("negative" in p for p in validate_structure(doc(total="-5")))
    assert any("quantity" in p for p in validate_structure(doc(items=[item(q="ten")])))
    assert any("date" in p.lower() for p in validate_structure(doc(document_date="20/10/2026")))
    assert validate_structure(doc()) == []


def test_missing_fields_reported():
    warnings = validate_document(doc(document_id=None, vendor=None))
    assert any("Document ID" in w for w in warnings) and any("Vendor" in w for w in warnings)


def test_validation_does_not_modify_data():
    d = doc(items=[item(t="1.00")])
    before = repr(d)
    validate_document(d)
    assert repr(d) == before
