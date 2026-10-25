# Intelligent Procurement Verification and Supply Risk Prediction

A Streamlit app that

1. **reads** purchase orders, goods receipt notes (GRNs) and supplier invoices (PDF or image),
2. **matches** each invoice against its PO and receipts (three-way match) and flags discrepancies and duplicates,
3. **predicts** which open orders are likely to arrive late or incomplete, with the reasons,
4. **estimates** which materials may run short, and
5. **ranks** everything as alerts with a suggested follow-up.

It is a decision-support prototype. Extraction is rule-based and predictions come from a small
model, so people should review what it shows. Demo data is synthetic.

## Quick start

```
pip install -r requirements.txt
streamlit run app.py
```

Then open **Overview** and press **Load demo data**. Every page now has content.
(`paddleocr` and `paddlepaddle` in `requirements.txt` are only needed for scans and photos.
Digital PDFs work without them; remove those two lines if installation is a problem.)

## Gemini (reads photos and scans)

1. Get a key from Google AI Studio.
2. Streamlit Cloud: **App settings → Secrets**, add

```
GEMINI_API_KEY = "your-key"
# GEMINI_MODEL = "gemini-2.5-flash"   # optional; change if Google retires or renames the model
```

   Locally: put the same lines in `.streamlit/secrets.toml`, or set the `GEMINI_API_KEY` environment variable.
3. On **Extract documents**, switch on *Read with Gemini*. The flow per file is:
   text first (PyMuPDF for digital PDFs, PaddleOCR for photos and scans) -> Gemini turns that text into the
   app's JSON format (and detects PO / GRN / invoice with *Auto-detect*) -> you review -> save for matching.
   If Gemini is busy it is retried, then a backup model is tried; if it still fails, the rule-based extractor
   works on the same text. If no text could be read at all, Gemini reads the image directly.

Upload several files at once; they stay in one batch (adding more never removes earlier ones), can be
reviewed and saved together (PO, then GRN, then invoice), and the match result for the batch is shown right away.
Note: the extracted text (or the image, in the last-resort case) is sent to Gemini and leaves your machine.

## Try the whole chain with real files

`sample_docs/` has four synthetic PDFs: a PO, a GRN and two invoices.

1. **Extract documents** → type **Purchase Order** → upload `PO-5001.pdf` → *Extract* → *Edit and save* → **Save to database**.
2. Repeat with **Delivery Receipt / GRN** → `GRN-6001.pdf`, then **Invoice** → `INV-7001.pdf` and `INV-7002.pdf`.
3. Open **Three-way match**. `INV-7001` matches. `INV-7002` is held: price above the PO and more billed than received.

(Saving the same invoice twice is flagged as a duplicate, on purpose.)

## Pages

| Page | What it does |
|---|---|
| Overview | Counts, top alerts, invoice and risk summaries |
| Extract documents | Read a PDF/image, review and edit the JSON, save to the database |
| Three-way match | Status per invoice, discrepancy list, item-by-item comparison, adjustable tolerances |
| Delivery risk | Late / incomplete probability per open PO line, with the factors behind it |
| Shortage risk | Day-by-day stock projection under planned, expected and stress scenarios |
| Alerts | Ranked alerts with suggested follow-up; CSV download |
| Model performance | Baseline vs candidate models, confusion matrix, feature importance |
| Data | Demo data, CSV upload of history and inventory, reset |

## How it works

```
app.py                 page navigation
views/                 one file per page (UI only)
src/pipeline.py        file -> text (PyMuPDF, OCR fallback) -> fields
src/extract_fields.py  rule-based field and line-item extraction
src/ocr_reader.py      optional PaddleOCR
src/validation.py      arithmetic checks on one document
src/db.py              SQLite storage
src/matching.py        three-way match + duplicate detection
src/risk_model.py      late/incomplete classifiers + explanations
src/shortage.py        shortage projection
src/alerts.py          ranking and follow-up suggestions
src/analysis.py        runs the above over the database
src/demo_data.py       synthetic scenario
scripts/make_sample_pdfs.py   regenerates sample_docs/*.pdf
tests/                 95 automated tests
```

### Three-way match rules

All amounts use `Decimal`. Items are paired by name (exact first, then clearly similar; if two candidates
are about equally similar the item is left unmatched, never guessed). Tolerances: unit price 1 %,
quantity 0 units, amounts 0.01. Documents are never changed; differences are only reported.

| Finding | Severity |
|---|---|
| No PO, no receipt, duplicate invoice, item not on PO or not received, quantity above PO or above receipt, price above PO, supplier or currency mismatch | high |
| Price below PO, line or total arithmetic wrong, invoice dated before PO, possible duplicate (same supplier, date and total), receipt not on PO, over-delivery | medium |
| Invoice covers less than received | low |

Status: any high or medium → **Hold - review needed**; only low → **Matched with minor differences**; none → **Matched**.
A duplicate is flagged on the later-saved invoice only.

### Delivery-risk model

* Two classifiers per order line: **late** (actual lead time above promised) and **incomplete** (less than 98 % received).
* Features known at order time: quantity, value, promised lead time, month, weekday, the supplier's past late /
  incomplete rates and average delay, the material's past late rate, and order size versus typical.
* **No leakage.** Receipt dates and received quantities only create labels and supplier history built from
  orders that were already received *before* the order being scored (tested in `tests/test_risk_model.py`).
* Candidates: Baseline, Logistic Regression, Decision Tree, Random Forest. The best is chosen by time-ordered
  cross-validation on the earlier 75 % of history and scored on the latest 25 %.
* Overall risk = 1 − (1 − P(late)) × (1 − P(incomplete)), which assumes the two are independent.
  High ≥ 50 %, Medium ≥ 30 %.
* Explanations: each feature is swapped for its typical value in turn; the change in probability is its effect.

### Shortage estimate

Stock on day t = stock + deliveries arrived by day t − daily demand × t, for the next 30 days.
*Planned*: orders arrive on time and in full. *Expected*: arrival delayed by P(late) × the supplier's typical delay,
quantity cut by P(incomplete) × typical shortfall. *Stress*: every open order late and short by the typical amount.
High = stock-out in the plan or expected case; Medium = only in the stress case; Low = none.
Materials without stock or demand figures are shown as **Not assessed**.

## Using your own data

**Order history CSV** (needs 60+ completed orders): `po_number, supplier, material, order_date, quantity_ordered,
unit_price, promised_lead_days, actual_lead_days, quantity_received`. **Inventory CSV** (optional):
`material, stock, daily_demand`. Templates are on the **Data** page.

## Limits you should mention in your review

* Rule-based extraction works for common layouts, not all. Rows with extra columns are reported, not guessed.
* The demo history is synthetic, so model scores describe the generator, not real suppliers.
* Probabilities from a small model are not guarantees; thresholds are judgement calls.
* SQLite on free hosting can reset when the app restarts.
* English documents, one currency per document, no user login.

## Tests

```
pip install pytest
python -m pytest -q
```
