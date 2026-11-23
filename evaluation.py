import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from app import extract_pdf, segment_document
from conflict_detection import analyze_document, extract_notice_periods
from legal_extraction import (DURATION_RE, MONEY_RE, UNIT_DAYS, _to_int, extract_dates, extract_legal_information,
                              extract_money)
from rag_chat import answer_question, build_index, split_sentences

CACHE_DIR = Path("eval_cache")
RESULT_DIR = Path("eval_results")

# One natural-language question per CUAD category (CUAD category -> question)
QUESTIONS = {
    "Governing Law": "Which law governs this agreement?",
    "Termination For Convenience": "Can either party terminate this agreement for convenience?",
    "Notice Period To Terminate Renewal": "What notice period is required to terminate the renewal?",
    "Expiration Date": "When does this agreement expire?",
    "Renewal Term": "What is the renewal term of this agreement?",
    "Non-Compete": "Is there a non-compete restriction?",
    "Anti-Assignment": "Can the agreement be assigned or transferred to another party?",
    "Cap On Liability": "Is there a cap on liability?",
    "Exclusivity": "Is there an exclusivity obligation?",
    "Insurance": "What insurance must be maintained?",
    "Audit Rights": "Does a party have audit rights?",
    "Liquidated Damages": "Are there liquidated damages or penalties?",
}


# ----------------------------------------------------------------------------
# Person 6's metric functions (unchanged)
# ----------------------------------------------------------------------------
def precision_recall_f1(y_true, y_pred):
    true_positive = sum(actual and predicted for actual, predicted in zip(y_true, y_pred))
    false_positive = sum((not actual) and predicted for actual, predicted in zip(y_true, y_pred))
    false_negative = sum(actual and (not predicted) for actual, predicted in zip(y_true, y_pred))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"Precision": round(precision, 3), "Recall": round(recall, 3), "F1 Score": round(f1, 3)}


def retrieval_metrics(retrieved_ids, relevant_ids, k=3):
    relevant_ids = set(relevant_ids)
    top_k = retrieved_ids[:k]
    recall_at_k = len(set(top_k) & relevant_ids) / len(relevant_ids) if relevant_ids else 0
    rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid in relevant_ids), None)
    mrr = 1 / rank if rank else 0
    return {f"Recall@{k}": round(recall_at_k, 3), "MRR": round(mrr, 3)}


def prf(tp, fp, fn):
    """Precision/recall/F1 from counts."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f, 3), "TP": tp, "FP": fp, "FN": fn}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def overlaps(chunk_text, gold_text, threshold=0.6, min_tokens=4):
    """True if most of the smaller text's words appear in the other (tolerates OCR/line-break differences)."""
    c, g = set(norm(chunk_text).split()), set(norm(gold_text).split())
    if min(len(c), len(g)) < min_tokens:
        return norm(gold_text) in norm(chunk_text) and len(g) > 0
    return len(c & g) / min(len(c), len(g)) >= threshold


def index_pdfs(pdf_dir):
    found = {}
    for p in Path(pdf_dir).rglob("*.pdf"):
        found[p.stem.lower()] = p
    return found


def load_processed(pdf_path, ocr=False):
    """Run Person 2's pipeline on a PDF, cached on disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / (Path(pdf_path).stem[:120] + f"_{'ocr' if ocr else 'noocr'}.json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = Path(pdf_path).read_bytes()
    processed = segment_document(extract_pdf(data, Path(pdf_path).name, ocr))
    cache.write_text(json.dumps(processed, ensure_ascii=False), encoding="utf-8")
    return processed


def save_result(name, payload):
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cuad(path):
    """CUAD ships in SQuAD format: data[i].title, paragraphs[0].context, qas[].id = '<title>__<Category>'."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = []
    for entry in raw["data"]:
        qas = []
        for para in entry["paragraphs"]:
            for qa in para["qas"]:
                category = qa["id"].split("__")[-1].strip()
                qas.append({"category": category,
                            "answers": [a["text"] for a in qa.get("answers", []) if a.get("text", "").strip()]})
        docs.append({"title": entry["title"], "qas": qas})
    return docs


def select_docs(docs, pdfs, max_docs, seed):
    usable = [d for d in docs if d["title"].lower().removesuffix(".pdf") in pdfs]
    random.Random(seed).shuffle(usable)
    return usable[:max_docs]


def name_key(text):
    drop = {"inc", "ltd", "llc", "corp", "corporation", "co", "company", "pvt", "limited", "the", "plc", "llp"}
    return " ".join(w for w in norm(text).split() if w not in drop)


def name_match(a, b):
    na, nb = name_key(a), name_key(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    sa, sb = set(na.split()), set(nb.split())
    return len(sa & sb) / len(sa | sb) >= 0.6


# ----------------------------------------------------------------------------
# cuad: retrieval, QA, citations, legal information extraction, annotated-clause coverage
# ----------------------------------------------------------------------------
def check_citation(citation, processed, page_text, record_text):
    """Citation is accurate if the quoted evidence really is in the cited record AND on the cited page(s)."""
    ev = norm(citation["evidence_text"])
    if not ev:
        return False
    in_record = ev in norm(record_text.get(citation["record_id"], ""))
    pages = " ".join(page_text.get(p, "") for p in range(citation["page_start"], citation["page_end"] + 1))
    return in_record and ev in pages


def gold_dates(texts):
    out = set()
    for t in texts:
        for d in extract_dates(t):
            out.add(d["iso"] or norm(d["text"]))
    return out


def gold_notice_days(texts):
    out = set()
    for t in texts:
        for m in DURATION_RE.finditer(t):
            v = _to_int(m.group("paren") or m.group("num"))
            if v is not None:
                out.add(v * UNIT_DAYS[m.group("unit").lower()])
    return out


def cmd_cuad(args):
    docs, pdfs = load_cuad(args.cuad_json), index_pdfs(args.pdf_dir)
    chosen = select_docs(docs, pdfs, args.max_docs, args.seed)
    if not chosen:
        raise SystemExit("No CUAD contract titles matched a PDF in --pdf-dir. Check both paths.")
    print(f"Evaluating {len(chosen)} contracts (OCR {'on' if args.ocr else 'off'})")

    rk = {3: [], 5: []}
    mrr = []
    gold_q = supported = 0
    empty_q = abstained = 0
    unlocatable = 0
    cites = cites_ok = 0
    span_total = span_found = 0
    ext = {"dates": [0, 0, 0], "notice": [0, 0, 0], "parties": [0, 0, 0]}
    date_any_found = date_gold_total = 0
    details = []

    for n, doc in enumerate(chosen, 1):
        path = pdfs[doc["title"].lower().removesuffix(".pdf")]
        print(f"  [{n}/{len(chosen)}] {path.name}")
        processed = load_processed(path, args.ocr)
        index = build_index(processed)
        legal = extract_legal_information(processed)
        record_text = {r["record_id"]: r["text"] for r in processed["records"]}
        page_text = {p["page_number"]: norm(p["text"]) for p in processed["pages"]}
        by_cat = {qa["category"].lower(): qa["answers"] for qa in doc["qas"]}

        # ---- coverage of every annotated clause (any CUAD category) ----
        for qa in doc["qas"]:
            for g in qa["answers"]:
                span_total += 1
                span_found += any(overlaps(c["text"], g) for c in index.chunks)

        # ---- retrieval + QA + citations ----
        for category, question in QUESTIONS.items():
            gold = by_cat.get(category.lower())
            if gold is None:
                continue
            result = answer_question(index, question)
            if gold:
                relevant = [c["chunk_id"] for c in index.chunks if any(overlaps(c["text"], g) for g in gold)]
                if not relevant:
                    unlocatable += 1
                    details.append({"doc": path.name, "category": category, "note": "gold text not found in extracted text"})
                    continue
                retrieved = [h["chunk"]["chunk_id"] for h in index.search(question, top_k=10)]
                for k in rk:
                    rk[k].append(retrieval_metrics(retrieved, relevant, k)[f"Recall@{k}"])
                mrr.append(retrieval_metrics(retrieved, relevant, 5)["MRR"])
                gold_q += 1
                ok = result["answerable"] and any(
                    any(overlaps(c["evidence_text"], g) or overlaps(c["clause_text"], g) for g in gold)
                    for c in result["citations"])
                supported += bool(ok)
                details.append({"doc": path.name, "category": category, "answered": result["answerable"],
                                "evidence_supported": bool(ok),
                                "first_retrieved": retrieved[:3], "relevant": relevant[:5]})
            else:
                empty_q += 1
                abstained += not result["answerable"]
            for c in result["citations"]:
                cites += 1
                cites_ok += check_citation(c, processed, page_text, record_text)

        # ---- legal information extraction (Person 3) ----
        s = legal["document_summary"]
        g = gold_dates(by_cat.get("agreement date", []) + by_cat.get("effective date", []) + by_cat.get("expiration date", []))
        if g:
            pred = {d["iso"] or norm(d["text"]) for d in s["dates"]
                    if d["role"] in ("agreement_date", "effective_date", "start_date", "end_date")}
            alld = {d["iso"] or norm(d["text"]) for d in s["dates"]}
            ext["dates"][0] += len(pred & g); ext["dates"][1] += len(pred - g); ext["dates"][2] += len(g - pred)
            date_gold_total += len(g); date_any_found += len(g & alld)
        g = gold_notice_days(by_cat.get("notice period to terminate renewal", []))
        if g:
            pred = {x["days_approx"] for x in s["notice_periods"]}
            ext["notice"][0] += len(pred & g); ext["notice"][1] += len(pred - g); ext["notice"][2] += len(g - pred)
        gp = by_cat.get("parties", [])
        if gp:
            pn = [p["name"] for p in s["parties"]]
            tp = sum(any(name_match(a, b) for b in pn) for a in gp)
            ext["parties"][0] += tp
            ext["parties"][2] += len(gp) - tp
            ext["parties"][1] += sum(not any(name_match(a, b) for a in gp) for b in pn)

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    result = {
        "documents": len(chosen),
        "evidence_retrieval": {"questions": gold_q, "Recall@3": mean(rk[3]), "Recall@5": mean(rk[5]), "MRR": mean(mrr),
                               "gold_text_not_found_in_extraction": unlocatable},
        "question_answering": {
            "answerable_questions": gold_q,
            "evidence_support_rate": round(supported / gold_q, 3) if gold_q else None,
            "unanswerable_questions": empty_q,
            "correct_abstention_rate": round(abstained / empty_q, 3) if empty_q else None},
        "citation_accuracy": {"citations": cites, "accurate": cites_ok,
                              "accuracy": round(cites_ok / cites, 3) if cites else None},
        "annotated_clause_coverage": {"annotated_spans": span_total,
                                      "found_in_extracted_clauses": span_found,
                                      "recall": round(span_found / span_total, 3) if span_total else None},
        "legal_information_extraction": {
            "dates (role-labelled)": prf(*ext["dates"]),
            "dates recall ignoring role": round(date_any_found / date_gold_total, 3) if date_gold_total else None,
            "notice periods (precision is a lower bound)": prf(*ext["notice"]),
            "parties": prf(*ext["parties"]),
        },
        "details": details,
    }
    save_result("cuad", result)
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


# ----------------------------------------------------------------------------
# conflict: synthetic injection benchmark
# ----------------------------------------------------------------------------
DAY_RE = re.compile(r"(?:by|before|on or before|no later than)\s+the\s+(\d{1,2})(?:st|nd|rd|th)\s+day\b", re.I)


def mutate_notice(s):
    ps = extract_notice_periods(s)
    if not ps:
        return None
    p = ps[0]
    a, b = p["span"]
    new = p["value"] * 2 if p["value"] < 90 else max(1, p["value"] // 2)
    return s[:a] + f"{new} {p['unit']}s" + s[b:]


def mutate_amount(s):
    items, matches = extract_money(s), list(MONEY_RE.finditer(s))
    if not items or len(items) != len(matches):
        return None
    for item, m in zip(items, matches):
        if not item["purpose"]:
            continue
        group = "num" if m.group("num") else "num2"
        raw = m.group(group)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        new = f"{int(value * 2):,}" if "," in raw else str(int(value * 2))
        a, b = m.span(group)
        return s[:a] + new + s[b:]
    return None


def mutate_deadline(s):
    m = DAY_RE.search(s)
    if not m:
        return None
    day = int(m.group(1))
    new = day + 5 if day <= 23 else day - 5
    suffix = "th" if 10 <= new % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(new % 10, "th")
    return s[:m.start(1)] + f"{new}{suffix}" + s[m.end(1) + 2:]      # +2 skips the old st/nd/rd/th


def mutate_modality(s):
    m = re.search(r"\b(shall not|must not|may not)\b", s, re.I)
    if m:
        return s[:m.start()] + "may" + s[m.end():]
    m = re.search(r"\b(shall|must)\b(?!\s+not)", s, re.I)
    if m:
        return s[:m.end()] + " not" + s[m.end():]
    return None


MUTATORS = {"notice": (mutate_notice, "Notice"), "amount": (mutate_amount, "Amount"),
            "deadline": (mutate_deadline, "Deadline"), "modality": (mutate_modality, "Modality")}


def inject(processed, text):
    """Copy of the processed document with one extra clause appended on the last page."""
    last = processed["pages"][-1]["page_number"]
    meta = processed["document_metadata"]
    record = {"record_id": "INJ-0001", "record_type": "clause", "section_heading": "INJECTED TEST CLAUSE",
              "clause_number": "999.1", "text": text, "page_start": last, "page_end": last,
              "source_filename": meta["filename"], "extraction_method": "synthetic", "character_count": len(text),
              "source": {"filename": meta["filename"], "sha256": meta.get("sha256"), "page_start": last, "page_end": last}}
    return {**processed, "records": processed["records"] + [record]}


def cmd_conflict(args):
    rng = random.Random(args.seed)
    pdfs = sorted(Path(args.pdf_dir).rglob("*.pdf"))
    rng.shuffle(pdfs)
    pdfs = pdfs[:args.max_docs]
    if not pdfs:
        raise SystemExit("No PDFs found in --pdf-dir.")

    def run(processed):
        legal = extract_legal_information(processed)
        return analyze_document(legal, use_nli=args.nli)

    stats = defaultdict(lambda: {"cases": 0, "detected": 0, "type_correct": 0})
    clean_flags, clean_docs, examples = 0, 0, []
    misses = []
    for n, path in enumerate(pdfs, 1):
        print(f"  [{n}/{len(pdfs)}] {path.name}")
        try:
            processed = load_processed(path, args.ocr)
        except Exception as error:
            print("     skipped:", error)
            continue
        baseline = run(processed)
        clean_docs += 1
        clean_flags += baseline["summary"]["potential_conflict_count"]
        for issue in baseline["issues"][:2]:
            examples.append({"doc": path.name, "types": issue["conflict_types"],
                             "a": issue["clause_1"]["text"][:160], "b": issue["clause_2"]["text"][:160]})

        sentences = [s for r in processed["records"] if r["record_type"] != "section"
                     for s in split_sentences(r["text"]) if 40 <= len(s) <= 350]
        for kind, (mutate, label) in MUTATORS.items():
            options = [(s, mutate(s)) for s in sentences]
            options = [(s, m) for s, m in options if m and m != s]
            if not options:
                continue
            original, injected_text = rng.choice(options)
            report = run(inject(processed, injected_text))
            hit = [i for i in report["issues"] if "INJ-0001" in (i["clause_1"]["record_id"], i["clause_2"]["record_id"])]
            stats[kind]["cases"] += 1
            stats[kind]["detected"] += bool(hit)
            stats[kind]["type_correct"] += any(label in t for i in hit for t in i["conflict_types"])
            if not hit:
                misses.append({"doc": path.name, "kind": kind, "original": original[:200], "injected": injected_text[:200]})

    cases = sum(v["cases"] for v in stats.values())
    tp = sum(v["detected"] for v in stats.values())
    fn = cases - tp
    overall = prf(tp, clean_flags, fn)
    result = {
        "documents": clean_docs,
        "injected_conflicts": cases,
        "overall": overall,
        "type_correct_recall": round(sum(v["type_correct"] for v in stats.values()) / cases, 3) if cases else None,
        "per_type_recall": {k: round(v["detected"] / v["cases"], 3) for k, v in stats.items() if v["cases"]},
        "flags_per_clean_document": round(clean_flags / clean_docs, 2) if clean_docs else None,
        "nli_used": args.nli,
        "how_to_read": ("TP = injected conflict reported; FN = injected conflict missed; FP = every conflict flagged in the "
                        "UNMODIFIED contracts (assumed conflict-free - inspect 'clean_flag_examples' by hand, some may be "
                        "genuine). Injected clauses are edited copies of real sentences, so recall is optimistic."),
        "clean_flag_examples": examples[:15],
        "missed_examples": misses[:15],
    }
    save_result("conflict", result)
    print(json.dumps({k: v for k, v in result.items() if "examples" not in k and k != "how_to_read"}, indent=2))


# ----------------------------------------------------------------------------
# clauses: clause-number precision/recall against a small hand-corrected gold file
# ----------------------------------------------------------------------------
def clause_numbers(processed):
    nums = set()
    for r in processed["records"]:
        n = r.get("clause_number")
        if r["record_type"] == "clause" and n and re.match(r"^\d", n.strip()):
            nums.add(n.strip().rstrip("."))
    return nums


def cmd_clauses(args):
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    gold_path = Path(args.gold)
    if args.init:
        skeleton = {p.name: sorted(clause_numbers(load_processed(p, args.ocr))) for p in pdfs}
        gold_path.write_text(json.dumps(skeleton, indent=2), encoding="utf-8")
        print(f"Wrote {gold_path}. Open each PDF, then DELETE wrong numbers and ADD missing ones. Save and re-run without --init.")
        return
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    tp = fp = fn = 0
    per_doc = {}
    for name, nums in gold.items():
        pred = clause_numbers(load_processed(Path(args.pdf_dir) / name, args.ocr))
        g = {x.strip().rstrip(".") for x in nums}
        per_doc[name] = prf(len(pred & g), len(pred - g), len(g - pred))
        tp += len(pred & g); fp += len(pred - g); fn += len(g - pred)
    result = {"documents": len(gold), "overall": prf(tp, fp, fn), "per_document": per_doc}
    save_result("clauses", result)
    print(json.dumps(result, indent=2))


# ----------------------------------------------------------------------------
# table: merge everything into the project's results table
# ----------------------------------------------------------------------------
def cmd_table(_):
    def load(name):
        p = RESULT_DIR / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    cuad, conf, cl = load("cuad"), load("conflict"), load("clauses")
    fmt = lambda d: f"P={d['Precision']} / R={d['Recall']} / F1={d['F1']}"
    tbl = lambda v: v if v is not None else "n/a"
    rows = [
        ("Clause extraction", "Precision / Recall / F1",
         (fmt(cl["overall"]) + " (clause numbers, hand-corrected gold)") if cl else "not run (clauses command)"),
        ("Legal information extraction", "Precision / Recall / F1",
         "; ".join(f"{k}: {fmt(v)}" for k, v in cuad["legal_information_extraction"].items() if isinstance(v, dict))
         if cuad else "not run (cuad command)"),
        ("Conflict detection", "Precision / Recall / F1",
         (fmt(conf["overall"]) + f" on {conf['injected_conflicts']} injected conflicts") if conf else "not run (conflict command)"),
        ("Evidence retrieval", "Recall@k / MRR",
         (f"Recall@3={tbl(cuad['evidence_retrieval']['Recall@3'])}, Recall@5={tbl(cuad['evidence_retrieval']['Recall@5'])}, "
          f"MRR={tbl(cuad['evidence_retrieval']['MRR'])}") if cuad else "not run (cuad command)"),
        ("Question answering", "Correctness / evidence support",
         (f"evidence support={tbl(cuad['question_answering']['evidence_support_rate'])}, "
          f"correct abstention={tbl(cuad['question_answering']['correct_abstention_rate'])}") if cuad else "not run (cuad command)"),
        ("Source references", "Citation accuracy",
         tbl(cuad["citation_accuracy"]["accuracy"]) if cuad else "not run (cuad command)"),
    ]
    lines = ["| Component | Metric | Result |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "metrics_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pdf-dir", required=True)
        p.add_argument("--max-docs", type=int, default=20)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--ocr", action="store_true", help="enable OCR (slow; default off)")

    p = sub.add_parser("cuad"); common(p); p.add_argument("--cuad-json", required=True); p.set_defaults(fn=cmd_cuad)
    p = sub.add_parser("conflict"); common(p); p.add_argument("--nli", action="store_true", help="also use the NLI model (slower)"); p.set_defaults(fn=cmd_conflict)
    p = sub.add_parser("clauses"); common(p); p.add_argument("--gold", required=True); p.add_argument("--init", action="store_true"); p.set_defaults(fn=cmd_clauses)
    p = sub.add_parser("table"); p.set_defaults(fn=cmd_table)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
