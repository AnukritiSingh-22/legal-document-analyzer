
import json
import re
import sys
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# Number words / helpers
# ----------------------------------------------------------------------------
NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "forty-five": 45, "twenty-four": 24, "forty-eight": 48,
}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}  # approximate, for comparison only
MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
          "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")

# ----------------------------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------------------------
_NUMWORD = "|".join(sorted(map(re.escape, NUM_WORDS), key=len, reverse=True))

DURATION_RE = re.compile(
    rf"\b(?P<num>\d+|{_NUMWORD})(?:\s*\(\s*(?P<paren>\d+)\s*\))?\s*"
    r"(?P<daytype>calendar|business|working|clear)?\s*"
    r"(?P<unit>day|week|month|year)s?\b(?:['’]s?)?",
    re.IGNORECASE,
)

MONEY_RE = re.compile(
    r"(?P<cur>Rs\.?|INR|₹|USD|US\$|\$|EUR|€|GBP|£)\s?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?/-)?(?:\s?(?P<mult>lakhs?|lacs?|crores?|million|billion|thousand|k)\b)?"
    r"|(?P<num2>\d[\d,]*(?:\.\d+)?)\s?(?P<cur2>rupees|dollars|INR|USD)\b",
    re.IGNORECASE,
)
CUR_NORMAL = {"rs": "INR", "rs.": "INR", "inr": "INR", "₹": "INR", "rupees": "INR",
              "usd": "USD", "us$": "USD", "$": "USD", "dollars": "USD",
              "eur": "EUR", "€": "EUR", "gbp": "GBP", "£": "GBP"}
MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7,
        "million": 1e6, "billion": 1e9, "thousand": 1e3, "k": 1e3}

PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|per\s?cent|percent)", re.IGNORECASE)

DATE_RES = [
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:{MONTHS})\.?,?\s+\d{{4}}\b", re.I),
    re.compile(rf"\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]
DATE_ROLES = {
    "effective": "effective_date", "commenc": "start_date", "start": "start_date",
    "expir": "end_date", "terminat": "end_date", "end": "end_date",
    "due": "due_date", "payable": "due_date", "renew": "renewal_date",
    "dated": "agreement_date", "executed": "agreement_date", "signed": "agreement_date",
    "made on": "agreement_date", "entered into": "agreement_date",
}
MONEY_PURPOSES = ["salary", "rent", "deposit", "fee", "fees", "penalty", "damages", "bonus",
                  "compensation", "payment", "interest", "fine", "liquidated", "refund",
                  "expense", "reimburse", "stipend", "commission", "price", "charges", "cost", "liability", "allowance"]

GENERIC_PARTIES = [
    "Company", "Employee", "Employer", "Landlord", "Tenant", "Lessor", "Lessee", "Licensor",
    "Licensee", "Disclosing Party", "Receiving Party", "Client", "Customer", "Contractor",
    "Consultant", "Service Provider", "Vendor", "Supplier", "Buyer", "Seller", "Purchaser",
    "Party A", "Party B", "First Party", "Second Party", "Owner", "Borrower", "Lender",
    "Recipient", "Discloser", "Contractor",
]

# Defined terms:  (the "Company")  /  ("Employee")  /  "Confidential Information" means ...
PARTY_DEF_RE = re.compile(
    r"(?P<name>[A-Z][\w&.,'’\- ]{2,200}?)\s*,?\s*\(\s*(?:hereinafter\s+)?"
    r"(?:referred\s+to\s+as\s+|called\s+|the\s+)?(?:the\s+)?[\"“'‘](?P<alias>[^\"”'’]{2,40})[\"”'’]",
    re.IGNORECASE,
)
DEFINED_PAREN_RE = re.compile(r"\(\s*(?:the\s+)?[\"“](?P<term>[A-Z][^\"”]{1,50})[\"”]\s*\)")
DEFINED_MEANS_RE = re.compile(
    r"[\"“](?P<term>[A-Z][^\"”]{1,50})[\"”]\s+(?:shall\s+mean|means|refers\s+to|shall\s+refer\s+to|"
    r"has\s+the\s+meaning)\s+(?P<definition>[^.;]{5,300})",
    re.IGNORECASE,
)

# Modality (checked in this order)
MODALITY_RULES = [
    ("prohibition", re.compile(
        r"\b(shall\s+not|must\s+not|may\s+not|will\s+not|cannot|can\s+not|is\s+prohibited\s+from|"
        r"are\s+prohibited\s+from|is\s+not\s+(?:permitted|allowed)\s+to|shall\s+be\s+prohibited\s+from|"
        r"agrees\s+not\s+to|undertakes\s+not\s+to|shall\s+refrain\s+from)\b", re.I)),
    ("right", re.compile(
        r"\b(has\s+the\s+right\s+to|have\s+the\s+right\s+to|reserves\s+the\s+right\s+to|"
        r"is\s+entitled\s+to|are\s+entitled\s+to|shall\s+be\s+entitled\s+to)\b", re.I)),
    ("permission", re.compile(
        r"\b(may|is\s+permitted\s+to|are\s+permitted\s+to|is\s+allowed\s+to|are\s+allowed\s+to|"
        r"can|shall\s+be\s+free\s+to)\b", re.I)),
    ("obligation", re.compile(
        r"\b(shall|must|will|is\s+required\s+to|are\s+required\s+to|agrees\s+to|undertakes\s+to|"
        r"is\s+obligated\s+to|is\s+obliged\s+to|is\s+responsible\s+for|are\s+responsible\s+for|"
        r"needs\s+to|has\s+to|have\s+to)\b", re.I)),
]

CONDITION_RE = re.compile(
    r"\b(if|provided\s+that|provided\s+however|subject\s+to|in\s+the\s+event\s+(?:that|of)|"
    r"in\s+case|on\s+condition\s+that|where|whenever|upon\s+(?:the\s+)?(?:occurrence|receipt|termination|breach)|"
    r"in\s+the\s+case\s+of|as\s+soon\s+as)\b", re.I)
EXCEPTION_RE = re.compile(
    r"\b(unless|except(?:\s+(?:for|that|as))?|other\s+than|save\s+(?:for|that)|"
    r"notwithstanding|excluding|without\s+prejudice\s+to|apart\s+from)\b", re.I)
TIME_PHRASE_RE = re.compile(
    r"\b(within\s+[^,;.]{1,40}|no\s+later\s+than\s+[^,;.]{1,40}|on\s+or\s+before\s+[^,;.]{1,40}|"
    r"by\s+(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)?|first|last|end)\b[^,;.]{0,35}|"
    r"immediately|promptly|forthwith|without\s+delay|"
    r"(?:daily|weekly|monthly|quarterly|annually|yearly)|"
    r"(?:per|each|every)\s+(?:day|week|month|quarter|year)|"
    r"(?:for\s+a\s+period\s+of|for\s+a\s+term\s+of|for\s+a\s+duration\s+of)\s+[^,;.]{1,40}|"
    r"during\s+the\s+(?:term|period|employment)[^,;.]{0,30}|"
    r"after\s+(?:the\s+)?(?:termination|expiry|expiration)[^,;.]{0,30})", re.I)

NOTICE_CONTEXT_RE = re.compile(r"\bnotice\b|\bnotify\b|\bnotif", re.I)

ABBREVIATIONS = ["Rs.", "No.", "Inc.", "Ltd.", "Pvt.", "Co.", "e.g.", "i.e.", "etc.", "Mr.", "Mrs.",
                 "Ms.", "Dr.", "vs.", "Sr.", "Jr.", "Corp.", "St.", "Art.", "Sec.", "Cl."]


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def _to_int(token):
    token = token.lower()
    return int(token) if token.isdigit() else NUM_WORDS.get(token)


def _clean(text, limit=None):
    text = re.sub(r"\s+", " ", text or "").strip(" ,;:.-")
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text


def split_sentences(text):
    protected = text
    for ab in ABBREVIATIONS:
        protected = protected.replace(ab, ab.replace(".", "<DOT>"))
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)           # 1.2, 3.5
    protected = re.sub(r"(\(\w{1,4}\))\s*", r"\1 ", protected)
    protected = re.sub(r"\s+", " ", protected)  # join PDF line-wraps
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(\"“\d])", protected)
    return [_clean(p.replace("<DOT>", ".")) for p in parts if _clean(p)]


def _norm_date(raw):
    s = re.sub(r"(\d)(st|nd|rd|th)", r"\1", raw, flags=re.I)
    s = re.sub(r"\bday\s+of\s+", "", s, flags=re.I).replace(",", "").replace(".", "")
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d/%m/%Y", "%d-%m-%Y",
                "%d/%m/%y", "%d-%m-%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------------------
# Entity extractors (work on any text string)
# ----------------------------------------------------------------------------
def extract_durations(text):
    out = []
    for m in DURATION_RE.finditer(text):
        value = _to_int(m.group("paren") or m.group("num"))
        if value is None:
            continue
        unit = m.group("unit").lower()
        ctx = text[max(0, m.start() - 60): m.end() + 60]
        out.append({
            "text": _clean(m.group(0)),
            "value": value,
            "unit": unit + ("s" if value != 1 else ""),
            "day_type": (m.group("daytype") or "").lower() or None,
            "days_approx": value * UNIT_DAYS[unit],
            "is_notice_period": bool(NOTICE_CONTEXT_RE.search(ctx)),
        })
    return out


def extract_money(text):
    out = []
    for m in MONEY_RE.finditer(text):
        cur = (m.group("cur") or m.group("cur2") or "").lower()
        num = (m.group("num") or m.group("num2") or "").replace(",", "")
        try:
            value = float(num)
        except ValueError:
            continue
        mult = (m.group("mult") or "").lower()
        if mult:
            value *= MULT.get(mult, 1)
        ctx_start = max(0, m.start() - 70)
        ctx = text[ctx_start: m.end() + 70].lower()
        # nearest whole-word keyword wins (substring matching used to turn "current"/"different" into "rent")
        purpose, best_dist = None, None
        for p in MONEY_PURPOSES:
            for pm in re.finditer(rf"\b{re.escape(p)}", ctx):
                dist = abs(pm.start() - (m.start() - ctx_start))
                if best_dist is None or dist < best_dist:
                    purpose, best_dist = p, dist
        out.append({
            "text": _clean(m.group(0)),
            "currency": CUR_NORMAL.get(cur, cur.upper() or None),
            "value": value,
            "purpose": purpose,
        })
    return out


def extract_dates(text):
    out, seen = [], set()
    for rx in DATE_RES:
        for m in rx.finditer(text):
            if m.span() in seen:
                continue
            seen.add(m.span())
            ctx = text[max(0, m.start() - 50): m.start()].lower()
            role = next((r for k, r in DATE_ROLES.items() if k in ctx), None)
            out.append({"text": m.group(0), "iso": _norm_date(m.group(0)), "role": role})
    return out


def extract_percentages(text):
    return [_clean(m.group(0)) for m in PERCENT_RE.finditer(text)]


def extract_defined_terms(text):
    terms = {}
    for m in DEFINED_MEANS_RE.finditer(text):
        terms[m.group("term").strip()] = _clean(m.group("definition"), 250)
    for m in DEFINED_PAREN_RE.finditer(text):
        terms.setdefault(m.group("term").strip(), None)
    return terms


# ----------------------------------------------------------------------------
# Party detection
# ----------------------------------------------------------------------------
def detect_parties(records, use_spacy=False):
    """Look for 'X (the "Alias")' patterns in the opening of the document."""
    opening = re.sub(r"\s+", " ", " ".join(r["text"] for r in records[:25]))
    parties, aliases = [], []
    for m in PARTY_DEF_RE.finditer(opening):
        name = _clean(re.sub(r"^.*\b(?:between|and|by)\b\s+", "", m.group("name"), flags=re.I))
        alias = m.group("alias").strip()
        if re.search(r"\b(Pvt|Ltd|Inc|Corp|Co|LLP)$", name):
            name += "."
        if len(name) > 2 and alias.lower() not in {"agreement", "contract", "effective date"}:
            parties.append({"name": name, "alias": alias})
            aliases.append(alias)

    if use_spacy:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            known = {p["name"].lower() for p in parties}
            for ent in nlp(opening[:5000]).ents:
                if ent.label_ in ("ORG", "PERSON") and ent.text.lower() not in known:
                    parties.append({"name": _clean(ent.text), "alias": None, "source": "spacy_ner"})
                    known.add(ent.text.lower())
        except Exception:
            pass  # spaCy/model not installed: silently fall back to rules

    # Generic role names that appear in the text also count as party aliases
    full_text = " ".join(r["text"] for r in records)
    for g in GENERIC_PARTIES:
        if re.search(rf"\b{re.escape(g)}\b", full_text) and g not in aliases:
            aliases.append(g)
    return parties, sorted(set(aliases), key=len, reverse=True)


# ----------------------------------------------------------------------------
# Statement-level (obligation / right / permission / prohibition) extraction
# ----------------------------------------------------------------------------
def _find_modality(sentence):
    for label, rx in MODALITY_RULES:
        m = rx.search(sentence)
        if m:
            return label, m
    return None, None


def _find_party(before_text, aliases):
    best, best_pos = None, -1
    for a in aliases:
        for m in re.finditer(rf"\b{re.escape(a)}\b", before_text, flags=re.I):
            if m.start() > best_pos:
                best, best_pos = a, m.start()
    return best


def _clause_phrase(sentence, rx, limit=220):
    m = rx.search(sentence)
    if not m:
        return None
    tail = sentence[m.start():]
    end = re.search(r"[;]|(?<!\d),(?=\s+(?:the|each|and|then|such|it|he|she|they|[A-Z]))", tail[len(m.group(0)):])
    cut = len(m.group(0)) + end.start() if end else len(tail)
    return _clean(tail[:cut], limit)


def extract_statements(sentence, aliases):
    modality, m = _find_modality(sentence)
    if not m:
        return None
    before = sentence[:m.start()]
    after = sentence[m.end():]
    party = _find_party(before, aliases)
    if party is None and re.search(r"\b(each|both|either)\s+part(y|ies)\b|\bthe\s+parties\b", before, re.I):
        party = "Parties"

    action_cut = re.split(CONDITION_RE.pattern + "|" + EXCEPTION_RE.pattern + r"|;", after, maxsplit=1,
                          flags=re.I)[0]
    time_phrases = [_clean(t.group(0), 80) for t in TIME_PHRASE_RE.finditer(sentence)]
    durations = extract_durations(sentence)
    dates = extract_dates(sentence)

    return {
        "sentence": sentence,
        "party": party,
        "modality": modality,
        "modal_verb": _clean(m.group(0)),
        "action": _clean(action_cut, 200),
        "condition": _clause_phrase(sentence, CONDITION_RE),
        "exception": _clause_phrase(sentence, EXCEPTION_RE),
        "time": {
            "phrases": time_phrases,
            "durations": durations,
            "dates": [d["iso"] or d["text"] for d in dates],
        },
        "amounts": extract_money(sentence),
        "is_negated": modality == "prohibition",
    }


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def extract_legal_information(processed, use_spacy=False):
    records = processed.get("records", [])
    meta = processed.get("document_metadata", {})

    parties, aliases = detect_parties(records, use_spacy=use_spacy)

    # defined terms across the document (collected first so clause analysis can use them)
    defined_terms = {}
    for r in records:
        for term, definition in extract_defined_terms(r["text"]).items():
            if term not in defined_terms or (definition and not defined_terms[term]["definition"]):
                defined_terms[term] = {"definition": definition, "record_id": r["record_id"],
                                       "page": r["page_start"]}
    aliases = sorted(set(aliases) | {p["alias"] for p in parties if p.get("alias")}, key=len, reverse=True)

    clauses, all_statements = [], []
    doc_dates, doc_money, doc_notice, doc_pct = [], [], [], []

    for r in records:
        text = r["text"]
        source = {
            "record_id": r["record_id"],
            "record_type": r["record_type"],
            "clause_number": r.get("clause_number"),
            "section_heading": r.get("section_heading"),
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "filename": r.get("source_filename") or meta.get("filename"),
        }

        dates, money = extract_dates(text), extract_money(text)
        durations, pcts = extract_durations(text), extract_percentages(text)

        statements = []
        if r["record_type"] != "section":
            for sent in split_sentences(text):
                st = extract_statements(sent, aliases)
                if st:
                    st["source"] = source
                    statements.append(st)
                    all_statements.append(st)

        for d in dates:
            doc_dates.append({**d, "source": source})
        for a in money:
            doc_money.append({**a, "source": source})
        for d in durations:
            if d["is_notice_period"]:
                doc_notice.append({**d, "source": source})
        for p in pcts:
            doc_pct.append({"text": p, "source": source})

        clauses.append({
            **source,
            "text": text,
            "entities": {
                "dates": dates, "amounts": money, "durations": durations, "percentages": pcts,
            },
            "statements": statements,
            "has_legal_statement": bool(statements),
        })

    by_modality = {}
    for s in all_statements:
        by_modality[s["modality"]] = by_modality.get(s["modality"], 0) + 1

    return {
        "document_metadata": meta,
        "document_summary": {
            "parties": parties,
            "party_aliases": aliases,
            "defined_terms": defined_terms,
            "dates": doc_dates,
            "amounts": doc_money,
            "notice_periods": doc_notice,
            "percentages": doc_pct,
            "counts": {
                "records": len(records),
                "statements": len(all_statements),
                "by_modality": by_modality,
            },
        },
        "clauses": clauses,
        "statements": all_statements,
        "extraction_method": "rule-based" + (" + spaCy NER" if use_spacy else ""),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Automated extraction; outputs require human review and are not legal advice.",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python legal_extraction.py <person2_processed.json> [output.json]")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    result = extract_legal_information(data)
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace(".json", "_legal.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved {out}: {result['document_summary']['counts']}")
