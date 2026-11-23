
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 900          # long clauses are split into pieces of about this size
MIN_CHUNK_CHARS = 15           # tiny fragments are ignored
MIN_COVERAGE = 0.34            # share of the question's key terms that must appear in the evidence
MIN_SPARSE_SCORE = 0.06        # minimum keyword similarity
MIN_DENSE_SCORE = 0.45         # or minimum embedding similarity
MAX_EVIDENCE_SENTENCES = 3

DISCLAIMER = ("Answer copied from the document text. Automated output - please verify "
              "against the original contract. Not legal advice.")

STOPWORDS = set("""
a an the and or of to in on at by for with from as is are was were be been being this that these
those it its any all each such shall will would should can could may might must do does did not no
under per than then there their they them his her he she i you we our your what which who whom
whose when where why how about into over after before between within without if else also has have
had agreement contract document clause section say says said state states mention mentioned tell
explain please give show find
""".split())

# Words that mean roughly the same thing in contracts (used to widen keyword search)
TOPIC_GROUPS = [
    "terminate termination terminated cancel cancellation expiry expire end resign resignation",
    "notice notify notified notification written",
    "pay payment payable paid rent fee fees due salary invoice remuneration compensation",
    "deposit security refund refundable advance",
    "confidential confidentiality disclose disclosure nda secret proprietary",
    "liable liability damages indemnify indemnity penalty compensation",
    "dispute arbitration jurisdiction court governing law mediation",
    "renew renewal extend extension",
    "sublet sublease assign assignment transfer",
    "repair maintenance maintain damage",
    "leave vacation holiday absence",
    "non-compete noncompete competition solicit solicitation",
]


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def stem(word):
    """Very light stemmer so terminate/terminated/termination all become 'terminat'."""
    w = word.lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    for suffix in ("ing", "ion", "ed", "e"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            w = w[:-len(suffix)]
            break
    return w[:8]


def _words(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def keep_stems(text):
    """Stems of meaningful (non-stopword) words."""
    return [stem(w) for w in _words(text) if w not in STOPWORDS and len(w) > 1]


def analyzer(text):
    """Unigrams + bigrams of meaningful stems (used by TF-IDF)."""
    toks = keep_stems(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


GROUP_STEMS = [set(stem(w) for w in re.findall(r"[a-z0-9]+", g.lower())) for g in TOPIC_GROUPS]


def expand_query(question):
    """Add synonyms of any topic the question touches."""
    q_stems = set(keep_stems(question))
    extra = []
    for group in GROUP_STEMS:
        if q_stems & group:
            extra.extend(sorted(group))
    return question + " " + " ".join(extra)


ABBREVIATIONS = ["Rs.", "No.", "Inc.", "Ltd.", "Pvt.", "Co.", "e.g.", "i.e.", "etc.", "Mr.", "Mrs.",
                 "Ms.", "Dr.", "vs.", "Sr.", "Jr.", "Corp.", "St.", "Art.", "Sec.", "Cl."]


def split_sentences(text):
    protected = re.sub(r"\s+", " ", text or "").strip()
    for ab in ABBREVIATIONS:
        protected = protected.replace(ab, ab.replace(".", "<DOT>"))
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(\"\u201c\d])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


# ----------------------------------------------------------------------------
# Loading chunks from Person 2 / Person 3 JSON
# ----------------------------------------------------------------------------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _display_clause(number, last_main):
    """Person 2 stores sub-items as '(a)'. Turn them into '4.1(a)' using the last numbered clause."""
    if not number:
        return None
    number = number.strip()
    if re.match(r"^\d", number):
        return number.rstrip(".")
    m = re.match(r"^\(?([A-Za-z0-9ivxIVX]+)\)?$", number)
    if m and last_main:
        return f"{last_main}({m.group(1)})"
    return number


def build_reference(chunk):
    if chunk["page_start"] == chunk["page_end"]:
        parts = [f"Page {chunk['page_start']}"]
    else:
        parts = [f"Pages {chunk['page_start']}-{chunk['page_end']}"]
    if chunk.get("section_heading"):
        parts.append(f"Section: {chunk['section_heading']}")
    if chunk.get("display_clause"):
        parts.append(f"Clause {chunk['display_clause']}")
    elif not chunk.get("section_heading"):
        parts.append(chunk["record_id"])
    return " | ".join(parts)


def make_chunks(data):
    """Turn the records into searchable chunks that keep their source metadata."""
    records = data.get("records") or data.get("clauses") or []
    meta = data.get("document_metadata", {})
    chunks = []
    last_main = None
    last_section = None

    for r in records:
        if r.get("record_type") == "section":
            last_section = r.get("section_heading") or r.get("text")
            last_main = None
            continue

        text = re.sub(r"\s+", " ", r.get("text") or "").strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue

        number = r.get("clause_number")
        if number and re.match(r"^\d", number.strip()):
            last_main = number.strip().rstrip(".")
        display = _display_clause(number, last_main)

        heading = r.get("section_heading") or last_section
        page_start = r.get("page_start", r.get("page", 1))
        page_end = r.get("page_end", page_start)
        filename = r.get("source_filename") or r.get("filename") or meta.get("filename")

        # split long clauses into pieces (keeps sentence boundaries)
        pieces, current = [], ""
        for sent in split_sentences(text):
            if current and len(current) + len(sent) > MAX_CHUNK_CHARS:
                pieces.append(current)
                current = sent
            else:
                current = (current + " " + sent).strip()
        if current:
            pieces.append(current)

        for i, piece in enumerate(pieces):
            chunk = {
                "chunk_id": r["record_id"] if len(pieces) == 1 else f"{r['record_id']}#{i + 1}",
                "record_id": r["record_id"],
                "record_type": r.get("record_type"),
                "clause_number": number,
                "display_clause": display,
                "section_heading": heading,
                "page_start": page_start,
                "page_end": page_end,
                "filename": filename,
                "text": piece,
                "clause_text": text,
            }
            chunk["reference"] = build_reference(chunk)
            chunks.append(chunk)
    return chunks


# ----------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------
_EMBEDDER = None


def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
        except Exception:
            _EMBEDDER = False
    return _EMBEDDER or None


class DocumentIndex:
    def __init__(self, data, use_embeddings=True):
        self.metadata = data.get("document_metadata", {})
        self.chunks = make_chunks(data)
        self.chunk_stems = [set(keep_stems(c["text"] + " " + (c.get("section_heading") or ""))) for c in self.chunks]
        self.vectorizer = None
        self.matrix = None
        self.embeddings = None
        self.method = "empty"

        if not self.chunks:
            return

        # What gets searched: heading + clause number + text
        self.search_texts = [
            f"{c.get('section_heading') or ''} {c.get('display_clause') or ''} {c['text']}"
            for c in self.chunks
        ]
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.search_texts)
        self.method = "TF-IDF keyword search"

        if use_embeddings:
            model = _load_embedder()
            if model is not None:
                self.embeddings = model.encode(self.search_texts, normalize_embeddings=True,
                                               show_progress_bar=False)
                self.method = "Hybrid (embeddings + TF-IDF)"

    # -- scoring ---------------------------------------------------------------
    def scores(self, question):
        q_vec = self.vectorizer.transform([expand_query(question)])
        sparse = (self.matrix @ q_vec.T).toarray().ravel()
        dense = np.zeros(len(self.chunks))
        if self.embeddings is not None:
            q_emb = _load_embedder().encode([question], normalize_embeddings=True,
                                            show_progress_bar=False)[0]
            dense = self.embeddings @ q_emb
            combined = 0.5 * dense + 0.5 * np.minimum(1.0, sparse * 1.5)
        else:
            combined = sparse
        return combined, sparse, dense

    def search(self, question, top_k=5):
        if not self.chunks:
            return []
        combined, sparse, dense = self.scores(question)
        order = np.argsort(-combined)[:top_k]
        results = []
        for idx in order:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(combined[idx]),
                "sparse": float(sparse[idx]),
                "dense": float(dense[idx]),
                "index": int(idx),
            })
        return results

    def coverage(self, question, stems):
        """Share of the question's key terms present in the given stem set (synonyms count half)."""
        key = {k for k in set(keep_stems(question)) if len(k) > 2}
        if not key:
            return 0.0
        hit = 0.0
        for k in key:
            if k in stems:
                hit += 1.0                      # exact word found
                continue
            for group in GROUP_STEMS:
                if k in group and (group & stems):
                    hit += 0.5                  # only a synonym found
                    break
        return hit / len(key)


def build_index(data, use_embeddings=True):
    """data = Person 2's processed JSON dict (or Person 3's output dict)."""
    return DocumentIndex(data, use_embeddings=use_embeddings)


# ----------------------------------------------------------------------------
# Answering
# ----------------------------------------------------------------------------
def _best_sentence(index, question, chunk):
    sentences = split_sentences(chunk["text"])
    if not sentences:
        return chunk["text"], 0.0
    q_vec = index.vectorizer.transform([expand_query(question)])
    s_mat = index.vectorizer.transform(sentences)
    sims = (s_mat @ q_vec.T).toarray().ravel()
    best = int(np.argmax(sims))
    return sentences[best], float(sims[best])


def _citation(chunk, evidence, score):
    return {
        "reference": chunk["reference"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "section_heading": chunk["section_heading"],
        "clause_number": chunk["display_clause"],
        "record_id": chunk["record_id"],
        "chunk_id": chunk["chunk_id"],
        "filename": chunk["filename"],
        "evidence_text": evidence,
        "clause_text": chunk["clause_text"],
        "score": round(score, 4),
    }


def answer_question(index, question, top_k=5):
    question = (question or "").strip()
    base = {
        "question": question,
        "retrieval_method": index.method,
        "disclaimer": DISCLAIMER,
    }

    if not question:
        return {**base, "answerable": False, "answer": "Please type a question.",
                "confidence": "none", "citations": [], "closest_matches": []}

    if not index.chunks:
        return {**base, "answerable": False,
                "answer": "No usable text was found in this document, so I cannot answer.",
                "confidence": "none", "citations": [], "closest_matches": []}

    if not keep_stems(question):
        return {**base, "answerable": False,
                "answer": "I could not find a topic in that question. Please ask about a specific "
                          "term, for example notice period, rent, termination or confidentiality.",
                "confidence": "none", "citations": [], "closest_matches": []}

    hits = index.search(question, top_k=top_k)

    supported = []
    for h in hits:
        cov = index.coverage(question, index.chunk_stems[h["index"]])
        h["coverage"] = cov
        passes_score = h["sparse"] >= MIN_SPARSE_SCORE or h["dense"] >= MIN_DENSE_SCORE
        if cov >= MIN_COVERAGE and passes_score:
            supported.append(h)

    closest = [_citation(h["chunk"], h["chunk"]["text"][:300], h["score"]) for h in hits[:3]]

    if not supported:
        return {**base, "answerable": False,
                "answer": "The document does not contain enough information to answer this question.",
                "confidence": "none", "citations": [], "closest_matches": closest}

    # Pick the best sentence from each supported chunk
    picked = []
    for h in supported:
        sentence, s_score = _best_sentence(index, question, h["chunk"])
        picked.append((h, sentence, s_score))
    picked.sort(key=lambda x: (x[2], x[0]["score"]), reverse=True)

    top_sentence_score = picked[0][2]
    citations, lines, seen = [], [], set()
    for h, sentence, s_score in picked:
        if len(citations) >= MAX_EVIDENCE_SENTENCES:
            break
        if sentence in seen:
            continue
        if citations and s_score < 0.7 * top_sentence_score:
            continue
        seen.add(sentence)
        citations.append(_citation(h["chunk"], sentence, h["score"]))
        lines.append(f"{sentence} [{len(citations)}]")

    best_cov = max(h["coverage"] for h in supported)
    confidence = "high" if best_cov >= 0.67 else "medium" if best_cov >= 0.5 else "low"

    answer = "According to the document: " + " ".join(lines)
    return {**base, "answerable": True, "answer": answer, "confidence": confidence,
            "citations": citations, "closest_matches": []}


def ask(data_or_path, question):
    """One-call helper: ask("outputs/file_processed.json", "What is the rent?")"""
    data = load_json(data_or_path) if isinstance(data_or_path, (str, Path)) else data_or_path
    return answer_question(build_index(data), question)


# ----------------------------------------------------------------------------
# Terminal use
# ----------------------------------------------------------------------------
def print_result(result):
    print()
    print("Q:", result["question"])
    print("A:", result["answer"])
    if result["answerable"]:
        print(f"Confidence: {result['confidence']}")
        for i, c in enumerate(result["citations"], 1):
            print(f"  [{i}] {c['reference']}")
            print(f"      \"{c['evidence_text']}\"")
    elif result["closest_matches"]:
        print("Closest (not sufficient):")
        for c in result["closest_matches"]:
            print(f"  - {c['reference']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_chat.py <processed.json> [\"your question\"]")
        sys.exit(1)

    document = load_json(sys.argv[1])
    idx = build_index(document)
    print(f"Indexed {len(idx.chunks)} chunks using: {idx.method}")

    if len(sys.argv) > 2:
        print_result(answer_question(idx, " ".join(sys.argv[2:])))
    else:
        print("Type a question (or 'exit').")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", ""):
                break
            print_result(answer_question(idx, q))
