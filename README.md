# AI Legal Document Analyzer – integrated project

Upload a contract PDF → clauses with page numbers → parties / dates / amounts / obligations → **potential conflicts** with evidence → chat with the contract (answers cite page / section / clause).

```
PDF ─► app.py (P2)  ─► legal_extraction.py (P3) ─► conflict_detection.py (P4) ─► main_app.py (UI, P6)
        pages, clauses    parties, dates, amounts,     rules + embeddings + NLI        summary, tables, conflicts,
                          statements (modality…)                                       page highlighting
                      └► rag_chat.py (P5) ─► chat tab (chat_app.py helpers)
pipeline.py = glue (runs everything)     evaluation.py = metrics for the results table
```

## 1. Install (once)

You need Python 3.10–3.12 and internet for the first run (models download automatically).

```
cd legal_project
python3 -m venv venv
venv\Scripts\activate            (Mac/Linux:  source venv/bin/activate)
pip install -r requirements.txt
```

* The first analysis downloads two models from Hugging Face (MiniLM embeddings ≈ 90 MB, DeBERTa NLI a few hundred MB). Afterwards they are cached.
* **OCR is optional** – only needed for scanned PDFs. Install Tesseract (Windows installer: "UB Mannheim Tesseract"), tick *Add to PATH*. If you skip it, untick "OCR" in the app.
* If `torch`/`transformers` fail to install, the app still works: it falls back to TF-IDF similarity and skips NLI (the sidebar/caption tells you). Rule-based conflict detection is unaffected.

## 2. Check that everything works (2 minutes)

```
python3 -m streamlit run main_app.py
```
1. Sidebar → upload `sample_contracts/lease_conflicts.pdf` → pick *Lease / rental agreement* → **Analyse document**.
2. **Potential Conflicts** tab should list (at least) 4 issues: rent day 5th vs 10th, deposit Rs 50,000 vs 75,000, notice 30 vs 60 days, "shall not sublet" vs "may sublet". Tick *Show this page in the PDF* to see the clause highlighted.
3. **Chat** tab → click "What is the notice period for termination?" → answer cites clauses 3.1 and 4.2.
4. `employment_conflicts.pdf` → 2 issues (salary 80,000 vs 95,000; outside employment prohibited vs freelance allowed). `nda_clean.pdf` → 0 issues.

With the NLI model on you may see a few extra "Possible Semantic Contradiction" flags – that is expected; they are labelled "requires review".

Each run also saves `outputs/<name>_processed.json`, `_legal.json`, `_conflicts.json`.

## 3. Get the dataset (CUAD)

CUAD = 510 real commercial contracts, 41 expert-annotated clause types, **PDFs included**.

1. Download `CUAD_v1.zip` (~106 MB): https://zenodo.org/record/4595826
2. Unzip. You need two things from it: the **`full_contract_pdf`** folder and the annotation **`.json` file** (named `CUAD_v1.json` / `CUADv1.json`).
3. Put them anywhere, e.g. `C:\data\CUAD_v1\`.

## 4. Produce the metrics (run from the project folder, venv active)

```
python evaluation.py cuad     --cuad-json C:\data\CUAD_v1\CUAD_v1.json --pdf-dir C:\data\CUAD_v1\full_contract_pdf --max-docs 20
python evaluation.py conflict --pdf-dir C:\data\CUAD_v1\full_contract_pdf --max-docs 20
python evaluation.py clauses  --pdf-dir gold_pdfs --gold gold_clauses.json --init
   (open gold_clauses.json, fix the clause-number lists by looking at the 5 PDFs, save)
python evaluation.py clauses  --pdf-dir gold_pdfs --gold gold_clauses.json
python evaluation.py table
```
Start with `--max-docs 5` to check paths, then go to 20–30. Processed PDFs are cached in `eval_cache/`; results land in `eval_results/` (`metrics_table.md` is your filled-in table; the JSON files contain per-question details for error analysis).

| Table row | Command | How it is measured |
|---|---|---|
| Clause extraction P/R/F1 | `clauses` | clause numbers found vs. a hand-corrected list for ~5 PDFs (you create it with `--init`). `cuad` also reports *annotated-clause coverage* |
| Legal information extraction P/R/F1 | `cuad` | dates, renewal-notice periods and party names vs. CUAD annotations |
| Conflict detection P/R/F1 | `conflict` | conflicts are **injected** into real contracts (edited copy of a real sentence: notice, amount, deadline, may/shall-not) and the detector must find them |
| Evidence retrieval Recall@k / MRR | `cuad` | 12 CUAD clause types as questions; relevant = chunk overlapping the annotated span |
| Question answering | `cuad` | evidence-support rate on answerable questions; correct "not enough information" on unanswerable ones |
| Source references | `cuad` | citation accurate if the quoted evidence is really inside the cited clause **and** on the cited page |

**Read these honestly in the report**
* No public dataset labels contradictions *inside one contract*, so conflict detection uses synthetic injection. Recall is optimistic (injected clauses are near-copies); say so. Precision counts every flag in the *unmodified* contracts as a false alarm – skim `clean_flag_examples` in `eval_results/conflict.json`; some may be genuine conflicts, which you can report separately.
* Notice-period precision is a lower bound: CUAD only annotates the *renewal* notice period, so other valid notice periods count as false positives.
* CUAD PDFs are mostly text-based; OCR is off by default. If `gold_text_not_found_in_extraction` is high, extraction (not retrieval) is the problem – good material for error analysis.
* Optional second dataset for NDAs: ContractNLI (stanfordnlp.github.io/contract-nli).

## 5. What was changed in your teammates' files

**`app.py` (Person 2)** – UI moved into `main()`; PDF bytes kept in session state. Two bug fixes in the segmenter:
* A numbered clause whose first line wraps (`1.1 The Receiving Party shall not disclose … to any`) was classified as an ALL-CAPS *section heading* because the heading regex was case-insensitive. Those clauses vanished for everyone downstream (on the NDA test only 3 of 7 clauses survived). Now only `SECTION/ARTICLE/CHAPTER` is case-insensitive.
* `Section 2.1 Payment. The Company shall pay…` (long run-in headings) is kept as text instead of being swallowed as a heading.

**`legal_extraction.py` (Person 3)** – (1) an amount's *purpose* is now the nearest whole-word keyword (it used substring matching, so "current"/"different" became "rent"); (2) "made on" / "entered into" now label the agreement date.

**`conflict_detection.py` (Person 4)** – input is Person 3's output; embeddings computed once; a pair is compared if the clauses share a topic, the cheap rule checks run on all such pairs, NLI runs only on pairs above the similarity threshold. New rules the project summary asks for: prohibition vs permission (may / shall not), payment terms (30 vs 60 days), conflicting dates. Amounts are only compared when they are for the same purpose.

**`chat_app.py` / `rag_chat.py` (Person 5)** – only the `main()` wrapper in `chat_app.py`.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: fitz` | `pip install pymupdf` |
| `ModuleNotFoundError: streamlit` when running evaluation | `pip install streamlit` (the evaluation imports `app.py`) |
| "NLI model: not used" | `pip install transformers torch sentencepiece`; check internet for the first download |
| OCR warnings | Install Tesseract or untick OCR; text PDFs don't need it |
| `No CUAD contract titles matched a PDF` | `--pdf-dir` must be the `full_contract_pdf` folder (sub-folders are searched); `--cuad-json` the annotation file |
| Port already in use | `python -m streamlit run main_app.py --server.port 8502` |

## 7. Known limitations

Rule-based extraction (regex) misses unusual phrasing; clause segmentation depends on how the PDF's text flows; conflict flags are *potential* conflicts for human review, never legal conclusions; notice periods that legitimately differ by party are still flagged; jurisdiction-specific law is not modelled; chat answers are extractive (copied sentences), not generated.
