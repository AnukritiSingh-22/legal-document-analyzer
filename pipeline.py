
import re
from collections import Counter

from app import extract_pdf, segment_document
from conflict_detection import analyze_document
from legal_extraction import extract_legal_information
from rag_chat import make_chunks, split_sentences


def create_summary(processed, max_sentences=5):
    """Person 6's frequency-based extractive summary, now reading Person 2's records and
    using Person 5's page/section/clause references."""
    items = []
    for chunk in make_chunks(processed):
        for sentence in split_sentences(chunk["text"]):
            if 40 <= len(sentence) <= 400:
                items.append((sentence, chunk["reference"]))
    if not items:
        return []

    freq = Counter()
    for sentence, _ in items:
        freq.update(re.findall(r"[a-z]{3,}", sentence.lower()))

    def score(item):
        words = re.findall(r"[a-z]{3,}", item[0].lower())
        return sum(freq[w] for w in words) / len(words) if words else 0

    best = sorted(items, key=score, reverse=True)[:max_sentences]
    return [{"sentence": s, "reference": r} for s, r in best]


def analyse_processed(processed, similarity_threshold=0.55, use_nli=True):
    """Everything after PDF processing. Takes Person 2's processed dict."""
    legal = extract_legal_information(processed)
    conflicts = analyze_document(legal, similarity_threshold=similarity_threshold, use_nli=use_nli)
    return {
        "processed": processed,
        "legal": legal,
        "conflicts": conflicts,
        "summary": create_summary(processed),
    }


def run_pipeline(pdf_bytes, filename, enable_ocr=True, similarity_threshold=0.55, use_nli=True):
    """PDF bytes in, everything out."""
    extracted = extract_pdf(pdf_bytes, filename, enable_ocr)
    processed = segment_document(extracted)
    return analyse_processed(processed, similarity_threshold, use_nli)
