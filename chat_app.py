

import json

import streamlit as st

from rag_chat import answer_question, build_index

try:
    import fitz  # PyMuPDF (already installed for Person 2's app)
except Exception:
    fitz = None




# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing the document...")
def get_index(json_text):
    return build_index(json.loads(json_text))


def render_pdf_page(pdf_bytes, page_number, evidence):
    """Return PNG bytes of one page with the evidence text highlighted (None if unavailable)."""
    if fitz is None or not pdf_bytes:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number < 1 or page_number > len(doc):
            return None
        page = doc[page_number - 1]

        rects = page.search_for(evidence)
        if not rects:
            words = evidence.split()
            for i in range(0, len(words), 6):
                rects.extend(page.search_for(" ".join(words[i:i + 6])))
        for rect in rects:
            page.add_highlight_annot(rect)

        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def show_result(result, message_id, pdf_bytes):
    if result["answerable"]:
        st.markdown(result["answer"])
        st.caption(f"Confidence: {result['confidence']}  |  Retrieval: {result['retrieval_method']}")

        for i, c in enumerate(result["citations"], 1):
            with st.expander(f"[{i}] {c['reference']}"):
                st.markdown("**Supporting text**")
                st.info(c["evidence_text"])
                if c["clause_text"] != c["evidence_text"]:
                    st.markdown("**Full clause**")
                    st.write(c["clause_text"])
                st.caption(f"File: {c['filename']}  |  Record: {c['record_id']}")

                if pdf_bytes:
                    if st.checkbox("Show this page in the PDF", key=f"page_{message_id}_{i}"):
                        image = render_pdf_page(pdf_bytes, c["page_start"], c["evidence_text"])
                        if image:
                            st.image(image, caption=f"Page {c['page_start']}", width=700)
                        else:
                            st.warning("Could not render this page.")
    else:
        st.warning(result["answer"])
        if result["closest_matches"]:
            with st.expander("Closest clauses found (not enough to answer)"):
                for c in result["closest_matches"]:
                    st.markdown(f"- {c['reference']}")
                    st.caption(c["evidence_text"])

    st.caption(result["disclaimer"])


# ----------------------------------------------------------------------------
# Sidebar: load document
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Chat with the Contract", page_icon="💬", layout="wide")

    st.title("💬 Chat with the Contract")

    with st.sidebar:
        st.header("Document")
        json_file = st.file_uploader("Processed JSON (from Person 2)", type=["json"])
        pdf_file = st.file_uploader("Original PDF (optional, for page view)", type=["pdf"])

        if st.button("Clear chat"):
            st.session_state["messages"] = []

    json_text = None
    if json_file is not None:
        json_text = json_file.getvalue().decode("utf-8")
    elif "result" in st.session_state:
        json_text = json.dumps(st.session_state["result"], ensure_ascii=False)

    pdf_bytes = pdf_file.getvalue() if pdf_file is not None else None

    if json_text is None:
        st.info("Upload the processed JSON file from Person 2 in the sidebar to start.")
        st.stop()

    try:
        index = get_index(json_text)
    except Exception as error:
        st.error(f"Could not read this JSON file: {error}")
        st.stop()

    if not index.chunks:
        st.error("This file has no usable text. Please check the output of Person 2's step.")
        st.stop()

    st.caption(
        f"{index.metadata.get('filename', 'Document')} - {len(index.chunks)} searchable passages "
        f"- {index.method}"
    )

    # ----------------------------------------------------------------------------
    # Chat
    # ----------------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message_id, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                show_result(message["result"], message_id, pdf_bytes)

    question = st.chat_input("Ask about notice period, termination, rent, confidentiality...")

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        result = answer_question(index, question)
        st.session_state["messages"].append({"role": "assistant", "result": result})
        st.rerun()


if __name__ == "__main__":
    main()
