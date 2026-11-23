
import html
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="AI Legal Document Analyzer", page_icon="⚖️", layout="wide")

from chat_app import get_index, render_pdf_page, show_result      # noqa: E402  (Person 5's UI helpers)
from pipeline import run_pipeline                                  # noqa: E402
from rag_chat import answer_question, build_reference              # noqa: E402

SUGGESTED_QUESTIONS = {
    "Employment contract": ["What is my notice period?", "What are my obligations under this contract?",
                            "Can I work for another company?", "Where does the contract discuss confidentiality?"],
    "Lease / rental agreement": ["When must rent be paid?", "What is the security deposit?",
                                 "Who is responsible for maintenance?", "What is the notice period for termination?"],
    "NDA": ["What counts as confidential information?", "How long do the confidentiality obligations last?",
            "When does this agreement terminate?"],
    "General contract": ["When does this agreement terminate?", "What are the payment terms?",
                         "Which law governs this agreement?"],
}
SECTIONS = ["📋 Summary", "🔎 Extracted Information", "⚠️ Potential Conflicts", "📑 Clauses", "💬 Chat"]


def ref(source):
    """page / section / clause reference from a Person 3 'source' dict."""
    return build_reference({
        "page_start": source.get("page_start"), "page_end": source.get("page_end"),
        "section_heading": source.get("section_heading"), "display_clause": source.get("clause_number"),
        "record_id": source.get("record_id"),
    })


def save_outputs(analysis, filename):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    stem = Path(filename).stem
    for key, suffix in (("processed", "processed"), ("legal", "legal"), ("conflicts", "conflicts")):
        (out_dir / f"{stem}_{suffix}.json").write_text(
            json.dumps(analysis[key], indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.title("⚖️ AI Legal Document Analyzer")
st.caption("Academic prototype - automated analysis that requires human/legal review. Not legal advice.")

with st.sidebar:
    st.header("1. Upload")
    uploaded = st.file_uploader("Legal PDF", type=["pdf"])
    doc_type = st.selectbox("Document type", list(SUGGESTED_QUESTIONS))
    st.header("2. Settings")
    enable_ocr = st.checkbox("OCR for scanned pages", value=True,
                             help="Needs Tesseract installed. Text PDFs work without it.")
    use_nli = st.checkbox("Use NLI model for contradiction scores", value=True,
                          help="Downloads a small model on first use. If unavailable the rule checks still run.")
    sim_threshold = st.slider("Similarity threshold for NLI", 0.30, 0.90, 0.55, 0.05)
    analyse = st.button("Analyse document", type="primary", disabled=uploaded is None)

if analyse and uploaded is not None:
    try:
        with st.spinner("Processing PDF, extracting legal information, checking clauses..."):
            pdf_bytes = uploaded.getvalue()
            analysis = run_pipeline(pdf_bytes, uploaded.name, enable_ocr, sim_threshold, use_nli)
            out_dir = save_outputs(analysis, uploaded.name)
        st.session_state["analysis"] = analysis
        st.session_state["pdf_bytes"] = pdf_bytes
        st.session_state["pdf_name"] = uploaded.name
        st.session_state["chat_messages"] = []
        st.session_state["out_dir"] = str(out_dir)
        st.success("Analysis complete.")
    except Exception as error:
        st.error(f"Processing failed: {error}")

if "analysis" not in st.session_state:
    st.info("Upload a contract PDF in the sidebar and press **Analyse document**.")
    st.stop()

analysis = st.session_state["analysis"]
pdf_bytes = st.session_state.get("pdf_bytes")
processed, legal, conflicts = analysis["processed"], analysis["legal"], analysis["conflicts"]
records = processed["records"]
meta = processed["document_metadata"]
counts = legal["document_summary"]["counts"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pages", meta["page_count"])
c2.metric("Sections", sum(r["record_type"] == "section" for r in records))
c3.metric("Clauses / blocks", sum(r["record_type"] != "section" for r in records))
c4.metric("Legal statements", counts["statements"])
c5.metric("Potential conflicts", conflicts["summary"]["potential_conflict_count"])

for warning in processed.get("warnings", []):
    st.warning(warning)

view = st.radio("Section", SECTIONS, horizontal=True, label_visibility="collapsed")
st.divider()

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
if view == SECTIONS[0]:
    summary = legal["document_summary"]
    st.subheader("Key sentences")
    if analysis["summary"]:
        for item in analysis["summary"]:
            st.markdown(f"- {html.escape(item['sentence'])}  \n  <small>{html.escape(item['reference'])}</small>",
                        unsafe_allow_html=True)
    else:
        st.write("No extractable text found.")

    st.subheader("Parties")
    if summary["parties"]:
        st.dataframe([{"Name": p["name"], "Defined as": p.get("alias")} for p in summary["parties"]])
    else:
        st.write("No parties detected (looking for the pattern: Name (the \"Alias\")).")

    left, right = st.columns(2)
    with left:
        st.subheader("Dates")
        st.dataframe([{"Date": d["text"], "ISO": d["iso"], "Role": d["role"], "Where": ref(d["source"])}
                      for d in summary["dates"]] or [{"Date": "none found"}])
        st.subheader("Notice periods")
        st.dataframe([{"Notice": n["text"], "≈ Days": n["days_approx"], "Where": ref(n["source"])}
                      for n in summary["notice_periods"]] or [{"Notice": "none found"}])
    with right:
        st.subheader("Amounts")
        st.dataframe([{"Amount": a["text"], "Currency": a["currency"], "Value": a["value"],
                       "Purpose": a["purpose"], "Where": ref(a["source"])}
                      for a in summary["amounts"]] or [{"Amount": "none found"}])
        st.subheader("Defined terms")
        st.dataframe([{"Term": t, "Definition": v["definition"], "Page": v["page"]}
                      for t, v in summary["defined_terms"].items()] or [{"Term": "none found"}])

# ----------------------------------------------------------------------------
# Extracted information (Person 3's structured statements)
# ----------------------------------------------------------------------------
elif view == SECTIONS[1]:
    st.subheader("Obligations, rights, permissions and prohibitions")
    st.caption(f"Counts by type: {counts['by_modality']}")
    statements = legal["statements"]
    wanted = st.multiselect("Filter by type", sorted({s["modality"] for s in statements}),
                            default=sorted({s["modality"] for s in statements}))
    rows = [{
        "Where": ref(s["source"]), "Party": s["party"], "Type": s["modality"], "Action": s["action"],
        "Condition": s["condition"], "Exception": s["exception"],
        "Time": ", ".join(s["time"]["phrases"]), "Amounts": ", ".join(a["text"] for a in s["amounts"]),
    } for s in statements if s["modality"] in wanted]
    st.dataframe(rows or [{"Where": "no statements"}])
    st.download_button("Download legal information (JSON)",
                       json.dumps(legal, indent=2, ensure_ascii=False), file_name="legal_information.json",
                       mime="application/json")

# ----------------------------------------------------------------------------
# Potential conflicts (Person 4)
# ----------------------------------------------------------------------------
elif view == SECTIONS[2]:
    s = conflicts["summary"]
    st.subheader("Potential conflicts and inconsistencies - requires review")
    st.caption(
        f"{s['total_clauses']} clauses, {s['total_pairs']} possible pairs, {s['candidate_pairs']} candidate pairs "
        f"compared. Similarity: {s['similarity_method']}. "
        f"NLI model: {'used' if s['nli_used'] else 'not used (disabled or not installed)'}.")
    if not conflicts["issues"]:
        st.success("No potential conflicts were flagged. This does not guarantee the document is consistent.")
    for issue in conflicts["issues"]:
        a, b = issue["clause_1"], issue["clause_2"]
        title = f"{issue['issue_id']} | {' + '.join(issue['conflict_types'])} | {a['reference']}  ↔  {b['reference']}"
        with st.expander(title):
            st.warning(issue["reason"])
            left, right = st.columns(2)
            for col, clause, tag in ((left, a, "1"), (right, b, "2")):
                with col:
                    st.markdown(f"**Clause {tag}** - {clause['reference']}")
                    st.info(clause["text"])
                    if pdf_bytes and st.checkbox("Show this page in the PDF", key=f"cp_{issue['issue_id']}_{tag}"):
                        image = render_pdf_page(pdf_bytes, clause["page_start"], clause["text"])
                        if image:
                            st.image(image, caption=f"Page {clause['page_start']}", width=600)
                        else:
                            st.warning("Could not render this page.")
            nli = issue.get("nli_result")
            st.caption(f"Semantic similarity: {issue['semantic_similarity']}"
                       + (f"  |  NLI contradiction score: {nli['maximum_contradiction_score']}" if nli else ""))
    st.download_button("Download conflict report (JSON)", json.dumps(conflicts, indent=2, ensure_ascii=False),
                       file_name="conflict_report.json", mime="application/json")

# ----------------------------------------------------------------------------
# Clauses (Person 2)
# ----------------------------------------------------------------------------
elif view == SECTIONS[3]:
    st.subheader("Sections and clauses (with source metadata)")
    st.dataframe([{
        "ID": r["record_id"], "Type": r["record_type"], "Clause": r.get("clause_number"),
        "Section": r.get("section_heading"), "Pages": f"{r['page_start']}-{r['page_end']}",
        "Text": r["text"][:300],
    } for r in records])
    st.download_button("Download processed JSON", json.dumps(processed, indent=2, ensure_ascii=False),
                       file_name="processed_document.json", mime="application/json")
    st.caption(f"JSON files are also saved in: {st.session_state.get('out_dir', 'outputs')}")

# ----------------------------------------------------------------------------
# Chat (Person 5)
# ----------------------------------------------------------------------------
else:
    st.subheader("Chat with the contract")
    try:
        index = get_index(json.dumps(processed, ensure_ascii=False))
    except Exception as error:
        st.error(f"Could not build the search index: {error}")
        st.stop()
    st.caption(f"{len(index.chunks)} searchable passages - {index.method}")

    cols = st.columns(len(SUGGESTED_QUESTIONS[doc_type]))
    for i, q in enumerate(SUGGESTED_QUESTIONS[doc_type]):
        if cols[i].button(q, key=f"suggest_{i}"):
            st.session_state["pending_question"] = q

    messages = st.session_state.setdefault("chat_messages", [])
    for message_id, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                show_result(message["result"], message_id, pdf_bytes)

    question = st.chat_input("Ask about notice period, termination, rent, confidentiality...")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "result": answer_question(index, question)})
        st.rerun()
