import re
from functools import lru_cache
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from legal_extraction import DURATION_RE, NUM_WORDS, UNIT_DAYS, _to_int, extract_dates, extract_money
from rag_chat import STOPWORDS, _display_clause, _load_embedder, build_reference, stem

_money = lru_cache(maxsize=None)(extract_money)
_dates = lru_cache(maxsize=None)(extract_dates)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
SIMILARITY_THRESHOLD = 0.55   # NLI only runs on candidate pairs at least this similar
HIGH_SIMILARITY = 0.75        # pairs this similar are candidates even without a shared topic keyword
NLI_THRESHOLD = 0.80          # Person 4's final threshold
MAX_NLI_PAIRS = 300           # keeps CPU run time reasonable on long contracts
MIN_CLAUSE_CHARS = 25
STATUS_FLAG = "Potential Conflict - Requires Review"

# Person 4's topic filter + a few topics needed for the project's own examples
TOPIC_PATTERNS = {
    "rent_payment": r"\b(rent|rental payment|monthly payment)\b",
    "notice_period": r"\b(notice|termination|terminating|terminate)\b",
    "security_deposit": r"\b(security deposit|deposit)\b",
    "confidentiality": r"\b(confidential|confidentiality)\b",
    "payment_deadline": r"\b(payment deadline|due date|pay.*by)\b",
    # added
    "outside_work": r"\b(outside employment|other employment|freelanc\w*|moonlight\w*|other business|"
                    r"compet\w*|exclusiv\w*|outside work)\b",
    "salary": r"\b(salary|remuneration|compensation|wages?|stipend)\b",
    "payment": r"\b(pay|payment|payable|invoice|fees?)\b",
    "term_duration": r"\b(probation\w*|renew\w*|expir\w*|term of)\b",
    "liability": r"\b(liab\w*|indemn\w*|damages|penalt\w*)\b",
    "assignment": r"\b(sublet\w*|sublease\w*|assign\w*)\b",
    "maintenance": r"\b(maintenance|repair\w*)\b",
    "governing_law": r"\b(governing law|jurisdiction|arbitration)\b",
}

# Different words, same legal idea. Applied before comparing the actions of two statements.
CONCEPTS = [
    ("outside_work", r"outside (?:employment|work|business|activit\w+)|other (?:employment|business|work)|"
                     r"freelanc\w*|moonlight\w*|side (?:job|business)|part[- ]time (?:work|job)"),
    ("disclose", r"disclos\w*|reveal\w*|divulg\w*"),
    ("sublet", r"sublet\w*|sublease\w*|assign\w*"),
    ("terminate", r"terminat\w*|cancel\w*"),
]


# ----------------------------------------------------------------------------
# Models (lazy, optional)
# ----------------------------------------------------------------------------
_NLI = None


def _load_nli():
    global _NLI
    if _NLI is None:
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            _NLI = pipeline("text-classification", model=NLI_MODEL, tokenizer=NLI_MODEL,
                            device=device, top_k=None, truncation=True)
        except Exception:
            _NLI = False
    return _NLI or None


_EMB_CACHE = {}


def _embed(texts):
    model = _load_embedder()          # same MiniLM model Person 5 uses -> loaded only once
    if model is None:
        return None
    texts = list(texts)
    missing = [t for t in dict.fromkeys(texts) if t not in _EMB_CACHE]    # each distinct text is encoded once
    if missing:
        vectors = model.encode(missing, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        _EMB_CACHE.update(zip(missing, vectors))
    return np.array([_EMB_CACHE[t] for t in texts])


def similarity_matrix(texts):
    """Pairwise cosine similarity of all clauses. Returns (matrix, method_name)."""
    emb = _embed(texts)
    if emb is not None:
        return emb @ emb.T, "sentence-transformers all-MiniLM-L6-v2"
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
    return (tfidf @ tfidf.T).toarray(), "TF-IDF cosine (fallback)"


def calculate_similarity(clause_a, clause_b):
    """Person 4's original pairwise similarity."""
    emb = _embed([clause_a, clause_b])
    if emb is None:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform([clause_a, clause_b])
        return round(float((tfidf[0] @ tfidf[1].T).toarray()[0][0]), 4)
    return round(float(emb[0] @ emb[1]), 4)


def detect_contradiction(clause_a, clause_b):
    """Person 4's NLI check, run in both directions. Returns None if the NLI model is unavailable."""
    nli = _load_nli()
    if nli is None:
        return None
    scores_out = []
    for premise, hypothesis in ((clause_a, clause_b), (clause_b, clause_a)):
        output = nli({"text": premise, "text_pair": hypothesis})
        scores = output[0] if output and isinstance(output[0], list) else output
        scores_out.append({item["label"].lower(): float(item["score"]) for item in scores})
    contra = [s.get("contradiction", 0.0) for s in scores_out]
    return {
        "contradiction_score_forward": round(contra[0], 4),
        "contradiction_score_reverse": round(contra[1], 4),
        "maximum_contradiction_score": round(max(contra), 4),
        "status": "Requires Human Review",
    }


# ----------------------------------------------------------------------------
# Topic filter (Person 4)
# ----------------------------------------------------------------------------
def identify_topics(text):
    text = text.lower()
    return {t for t, p in TOPIC_PATTERNS.items() if re.search(p, text)}


def clauses_share_topic(clause_a, clause_b):
    return bool(identify_topics(clause_a) & identify_topics(clause_b))


# ----------------------------------------------------------------------------
# Rule checks. Each returns a dict with "conflict": True/False
# ----------------------------------------------------------------------------
def _ok():
    return {"conflict": False}


def detect_deadline_conflict(clause_a, clause_b):
    """Person 4: 'by the 5th day of each month' vs 'by the 10th day'. Also covers payment terms
    ('pay within 30 days' vs 'pay within 60 days'), which the project summary lists as an example."""
    day_re = re.compile(r"(?:by|before|on or before|no later than)\s+the\s+(\d{1,2})(?:st|nd|rd|th)\s+day\b", re.I)
    da, db = sorted(set(day_re.findall(clause_a))), sorted(set(day_re.findall(clause_b)))
    if da and db and da != db:
        return {"conflict": True, "conflict_type": "Potential Payment Deadline Inconsistency",
                "deadline_1": da, "deadline_2": db, "status": "Requires Human Review",
                "explanation": f"One clause sets a deadline on day {', '.join(da)} of the month, "
                               f"the other on day {', '.join(db)}."}

    pay_re = re.compile(r"(?:pay|payment|payable|invoice|due)\b[^.;]{0,80}?\bwithin\s+(?P<n>\d+|[a-z-]+)\s*"
                        r"(?:\(\s*\d+\s*\)\s*)?(?:calendar\s+|business\s+|working\s+)?(?P<u>day|week|month)s?", re.I)

    def pay_terms(text):
        out = set()
        for m in pay_re.finditer(text):
            n = _to_int(m.group("n"))
            if n is not None:
                out.add(n * UNIT_DAYS[m.group("u").lower()])
        return out

    pa, pb = pay_terms(clause_a), pay_terms(clause_b)
    if pa and pb and pa != pb:
        return {"conflict": True, "conflict_type": "Potential Payment Term Inconsistency",
                "term_days_1": sorted(pa), "term_days_2": sorted(pb), "status": "Requires Human Review",
                "explanation": f"Payment is due within {sorted(pa)} days in one clause but within "
                               f"{sorted(pb)} days in the other."}
    return _ok()


def detect_amount_conflict(clause_a, clause_b):
    """Person 4: different amounts. Now uses Person 3's money extractor (INR/USD/EUR/GBP, lakh/crore, words)
    and only compares amounts that are for the same purpose (rent vs deposit is not a conflict)."""
    ma, mb = _money(clause_a), _money(clause_b)
    for a in ma:
        for b in mb:
            same_currency = a["currency"] == b["currency"]
            purposes_compatible = a["purpose"] == b["purpose"] or a["purpose"] is None or b["purpose"] is None
            if same_currency and purposes_compatible and a["value"] != b["value"]:
                if a["purpose"] is None and b["purpose"] is None:
                    continue    # no idea what either amount is for: do not guess
                purpose = a["purpose"] or b["purpose"]
                return {"conflict": True, "conflict_type": "Potential Amount Inconsistency",
                        "amounts_clause_a": [a["text"]], "amounts_clause_b": [b["text"]],
                        "status": "Requires Human Review",
                        "explanation": f"Different amounts for '{purpose}': {a['text']} versus {b['text']}."}
    return _ok()


def extract_notice_periods(text):
    """Durations that are directly attached to the word 'notice' (e.g. '30 days written notice',
    'notice period of two months')."""
    out = []
    for m in DURATION_RE.finditer(text):
        value = _to_int(m.group("paren") or m.group("num"))
        if value is None:
            continue
        after = text[m.end(): m.end() + 30].lower()
        before = text[max(0, m.start() - 45): m.start()].lower()
        attached = re.match(r"^\W{0,3}(?:(?:prior|advance|written|termination)\s+)*notice\b", after) or re.search(
            r"\bnotice(?:\s+period)?(?:\s+(?:of|is|shall\s+be|will\s+be))?"
            r"(?:\s+(?:not\s+less\s+than|at\s+least|a\s+minimum\s+of|of))*\s*$", before)
        if attached:
            unit = m.group("unit").lower()
            out.append({"value": value, "unit": unit, "days": value * UNIT_DAYS[unit],
                        "text": m.group(0).strip(), "span": m.span()})
    return out


def detect_notice_conflict(clause_a, clause_b):
    pa, pb = extract_notice_periods(clause_a), extract_notice_periods(clause_b)
    da, db = {p["days"] for p in pa}, {p["days"] for p in pb}
    if da and db and da != db:
        fmt = lambda ps: ", ".join(f"{p['value']} {p['unit']}(s)" for p in ps)
        return {"conflict": True, "conflict_type": "Potential Notice Period Inconsistency",
                "period_1": [{"value": p["value"], "unit": p["unit"]} for p in pa],
                "period_2": [{"value": p["value"], "unit": p["unit"]} for p in pb],
                "status": "Requires Human Review",
                "explanation": f"Different notice periods: {fmt(pa)} versus {fmt(pb)}. "
                               f"(They may legitimately differ by party - please review.)"}
    return _ok()


def detect_date_conflict(clause_a, clause_b):
    """Same kind of date (effective / start / end / agreement / renewal) stated with different values."""
    da, db = _dates(clause_a), _dates(clause_b)
    for role in ("effective_date", "start_date", "end_date", "agreement_date", "renewal_date"):
        va = {d["iso"] or d["text"] for d in da if d["role"] == role}
        vb = {d["iso"] or d["text"] for d in db if d["role"] == role}
        if va and vb and va != vb:
            return {"conflict": True, "conflict_type": "Potential Date Inconsistency",
                    "dates_1": sorted(va), "dates_2": sorted(vb), "status": "Requires Human Review",
                    "explanation": f"The {role.replace('_', ' ')} is given as {sorted(va)} in one clause "
                                   f"and {sorted(vb)} in the other."}
    return _ok()


def _action_tokens(text, ignore):
    """Ordered content stems of an action phrase; synonyms are mapped to shared 'concept' tokens."""
    text = (text or "").lower()
    for name, pattern in CONCEPTS:
        text = re.sub(pattern, f" concept{name} ", text)
    words = re.findall(r"[a-z0-9]+", text)
    return [stem(w) if not w.startswith("concept") else w for w in words
            if w not in STOPWORDS and len(w) > 3 and w not in ignore]


def _party_compatible(p1, p2):
    if not p1 or not p2 or "Parties" in (p1, p2):
        return True
    return p1.lower() == p2.lower()


def detect_modality_conflict(statements_a, statements_b, ignore=()):
    """Uses Person 3's structured statements. Flags a prohibition in one clause against a
    permission / right / obligation in the other when the same party and a similar action are involved
    (e.g. 'Employee shall not undertake outside employment' vs 'Employee may undertake freelance work')."""
    ignore = {i.lower() for i in ignore}
    positive = {"permission", "right", "obligation"}
    for sa in statements_a:
        for sb in statements_b:
            ma, mb = sa["modality"], sb["modality"]
            opposed = (ma == "prohibition" and mb in positive) or (mb == "prohibition" and ma in positive)
            if not opposed or not _party_compatible(sa.get("party"), sb.get("party")):
                continue
            ta, tb = _action_tokens(sa.get("action"), ignore), _action_tokens(sb.get("action"), ignore)
            shared = set(ta) & set(tb)
            # a shared synonym concept only counts when it is part of the main action (first words of both phrases),
            # not when it merely appears later ("... from the date of disclosure")
            concept_shared = bool({t for t in ta[:3] if t.startswith("concept")} &
                                  {t for t in tb[:3] if t.startswith("concept")})
            same_head = bool(ta and tb and ta[0] == tb[0])
            # same legal idea (synonym concept) or same main verb plus another shared term.
            # "shall not disclose X" vs "shall use X" shares nouns but not the action, so it is not flagged.
            if not (concept_shared or (same_head and len(shared) >= 2)):
                continue
            prohibited, allowed = (sa, sb) if ma == "prohibition" else (sb, sa)
            note = ""
            if allowed.get("condition") or allowed.get("exception"):
                note = (f" The permitting statement is conditional ('{allowed.get('condition') or allowed.get('exception')}'), "
                        f"which may be an intended exception.")
            return {"conflict": True, "conflict_type": "Potential Modality Conflict (prohibition vs permission/obligation)",
                    "statement_1": sa["sentence"], "statement_2": sb["sentence"],
                    "shared_action_terms": sorted(shared)[:6], "status": "Requires Human Review",
                    "explanation": f"One statement prohibits an action ('{prohibited['modal_verb']}') while the other "
                                   f"{allowed['modality'] == 'obligation' and 'requires' or 'allows'} a similar action "
                                   f"('{allowed['modal_verb']}').{note}"}
    return _ok()


# ----------------------------------------------------------------------------
# Pair and document analysis
# ----------------------------------------------------------------------------
def analyze_clause_pair(clause_a, clause_b, statements_a=None, statements_b=None, ignore=(),
                        similarity=None, run_nli=True, nli_threshold=NLI_THRESHOLD):
    """Person 4's pair analysis (same output idea), extended with the modality/date rules."""
    findings = [
        detect_deadline_conflict(clause_a, clause_b),
        detect_amount_conflict(clause_a, clause_b),
        detect_notice_conflict(clause_a, clause_b),
        detect_date_conflict(clause_a, clause_b),
    ]
    if statements_a and statements_b:
        findings.append(detect_modality_conflict(statements_a, statements_b, ignore))
    rule_hits = [f for f in findings if f.get("conflict")]

    nli = detect_contradiction(clause_a, clause_b) if run_nli else None
    nli_concern = bool(nli) and nli["maximum_contradiction_score"] >= nli_threshold

    return {
        "clause_1": clause_a,
        "clause_2": clause_b,
        "semantic_similarity": similarity if similarity is not None else calculate_similarity(clause_a, clause_b),
        "nli_result": nli,
        "rule_based_findings": rule_hits,
        "status": STATUS_FLAG if (rule_hits or nli_concern) else "No Clear Conflict Detected",
    }


def prepare_clauses(legal):
    """Person 3's clauses -> flat list with a ready-made page/section/clause reference."""
    out, last_main = [], None
    for c in legal.get("clauses", []):
        if c.get("record_type") == "section":
            last_main = None
            continue
        number = c.get("clause_number")
        if number and re.match(r"^\d", number.strip()):
            last_main = number.strip().rstrip(".")
        item = {
            "record_id": c["record_id"],
            "clause_number": number,
            "display_clause": _display_clause(number, last_main),
            "section_heading": c.get("section_heading"),
            "page_start": c.get("page_start"),
            "page_end": c.get("page_end"),
            "text": re.sub(r"\s+", " ", c.get("text") or "").strip(),
            "statements": c.get("statements", []),
        }
        if len(item["text"]) >= MIN_CLAUSE_CHARS:
            item["reference"] = build_reference(item)
            out.append(item)
    return out


def _clause_view(c):
    return {"record_id": c["record_id"], "reference": c["reference"], "page_start": c["page_start"],
            "page_end": c["page_end"], "clause_number": c["display_clause"],
            "section_heading": c["section_heading"], "text": c["text"]}


def analyze_document(legal, similarity_threshold=SIMILARITY_THRESHOLD, nli_threshold=NLI_THRESHOLD,
                     use_nli=True, max_nli_pairs=MAX_NLI_PAIRS, high_similarity=HIGH_SIMILARITY):
    """
    legal = output of legal_extraction.extract_legal_information (needs "clauses" and "document_summary").
    Returns {"summary": {...counts, methods...}, "issues": [...], "note": ...}
    """
    clauses = prepare_clauses(legal)
    n = len(clauses)
    ignore = set(legal.get("document_summary", {}).get("party_aliases", []))
    total_pairs = n * (n - 1) // 2

    if n < 2:
        return {"summary": {"total_clauses": n, "total_pairs": 0, "candidate_pairs": 0, "analyzed_pairs": 0,
                            "skipped_pairs": 0, "potential_conflict_count": 0, "similarity_method": None,
                            "nli_used": False}, "issues": [],
                "note": "Not enough clauses to compare."}

    texts = [c["text"] for c in clauses]
    sim, sim_method = similarity_matrix(texts)
    topics = [identify_topics(t) for t in texts]
    norm = [re.sub(r"\W+", " ", t.lower()).strip() for t in texts]

    # Candidate pairs = clauses about the same topic (or very similar overall). The cheap rule checks run on all of
    # them because they need concrete evidence (different numbers / opposite modality). The slower and noisier NLI
    # model only runs on candidates that are also semantically similar (>= similarity_threshold).
    candidates = []
    for i, j in combinations(range(n), 2):
        if norm[i] == norm[j]:
            continue                                  # identical text repeated: not a conflict
        s = float(sim[i, j])
        if topics[i] & topics[j] or s >= high_similarity:
            candidates.append((s, i, j))
    candidates.sort(reverse=True)

    nli_ready = use_nli and _load_nli() is not None
    nli_runs = 0
    issues = []
    for s, i, j in candidates:
        a, b = clauses[i], clauses[j]
        run_nli = nli_ready and s >= similarity_threshold and nli_runs < max_nli_pairs
        nli_runs += run_nli
        report = analyze_clause_pair(
            a["text"], b["text"], a["statements"], b["statements"], ignore,
            similarity=round(s, 4), run_nli=run_nli, nli_threshold=nli_threshold)
        if report["status"] != STATUS_FLAG:
            continue

        types = [f["conflict_type"] for f in report["rule_based_findings"]]
        reasons = [f["explanation"] for f in report["rule_based_findings"]]
        nli = report["nli_result"]
        if nli and nli["maximum_contradiction_score"] >= nli_threshold:
            types.append("Possible Semantic Contradiction (NLI)")
            reasons.append(f"The NLI model rates these clauses as contradictory "
                           f"(score {nli['maximum_contradiction_score']}).")
        issues.append({
            "label": STATUS_FLAG,
            "conflict_types": types,
            "reason": " ".join(reasons),
            "clause_1": _clause_view(a),
            "clause_2": _clause_view(b),
            "semantic_similarity": report["semantic_similarity"],
            "nli_result": nli,
            "rule_based_findings": report["rule_based_findings"],
        })

    for k, issue in enumerate(issues, 1):
        issue["issue_id"] = f"CONF-{k:03d}"

    return {
        "summary": {
            "total_clauses": n,
            "total_pairs": total_pairs,
            "candidate_pairs": len(candidates),
            "analyzed_pairs": len(candidates),
            "skipped_pairs": total_pairs - len(candidates),
            "potential_conflict_count": len(issues),
            "similarity_method": sim_method,
            "nli_used": bool(nli_ready),
            "thresholds": {"similarity": similarity_threshold, "high_similarity": high_similarity,
                           "nli": nli_threshold},
        },
        "issues": issues,
        "note": "Potential conflicts only - automated output that requires human/legal review. Not legal advice.",
    }
