import streamlit as st
import fitz
import pdfplumber
import pytesseract
import hashlib
import io
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image


def extract_pdf(pdf_bytes, filename, enable_ocr=True):
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Please upload a PDF file.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError("The PDF is invalid or unreadable.") from error

    if doc.is_encrypted:
        doc.close()
        raise ValueError("Password-protected PDFs are not supported.")

    pages = []
    warnings = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(doc):
                page_number = i + 1
                text = page.get_text("text", sort=True).strip()
                method = "PyMuPDF"

                if len(text) < 30 and i < len(pdf.pages):
                    try:
                        alternate = (
                            pdf.pages[i].extract_text() or ""
                        ).strip()

                        if len(alternate) > len(text):
                            text = alternate
                            method = "pdfplumber"
                    except Exception:
                        pass

                if len(text) < 30 and enable_ocr:
                    try:
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(2, 2),
                            alpha=False
                        )

                        image = Image.open(
                            io.BytesIO(pix.tobytes("png"))
                        )

                        ocr_text = pytesseract.image_to_string(
                            image,
                            lang="eng"
                        ).strip()

                        if len(ocr_text) > len(text):
                            text = ocr_text
                            method = "Tesseract OCR"

                    except Exception:
                        warnings.append(
                            f"OCR failed on PDF page {page_number}. "
                            "Check whether Tesseract is installed."
                        )

                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "extraction_method": method,
                    "character_count": len(text),
                    "is_likely_blank": not bool(text.strip())
                })

    finally:
        doc.close()

    metadata = {
        "filename": Path(filename).name,
        "file_type": "application/pdf",
        "file_size_bytes": len(pdf_bytes),
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "page_count": len(pages)
    }

    return {
        "metadata": metadata,
        "pages": pages,
        "warnings": warnings
    }


# INTEGRATION FIX: IGNORECASE used to apply to the numbered-heading branch too, so a wrapped clause such as
# "1.1 The Receiving Party shall not disclose ..." (no full stop yet) was mistaken for a heading and lost.
# Now only the SECTION/ARTICLE/CHAPTER words are case-insensitive; numbered headings must be ALL CAPS.
SECTION_PATTERN = re.compile(
    r"^\s*(?:(?i:(?:SECTION|ARTICLE|CHAPTER)\s+[\w.-]+"
    r"(?:\s*[:.-]\s*|\s+).+)|"
    r"\d+(?:\.\d+){0,2}\s+[A-Z][A-Z0-9 ,/&()'’:-]{2,})\s*$"
)

CLAUSE_PATTERN = re.compile(
    r"^\s*("
    r"\d+(?:\.\d+){1,5}"
    r"|\d+\.(?!\d)"
    r"|\([a-zA-Z0-9ivxIVX]+\)"
    r"|[a-zA-Z]\)"
    r")\s+(\S.*)$"
)

SECTION_WORD_PATTERN = re.compile(
    r"^\s*(SECTION|ARTICLE|CHAPTER)\s+[\w.-]+",
    re.IGNORECASE
)


def segment_document(extracted):
    records = []
    current = None
    current_section = None
    clause_count = 0
    section_count = 0

    def save_current():
        nonlocal current

        if current is not None:
            full_text = "\n".join(
                current.pop("text_parts")
            ).strip()

            if full_text:
                current["text"] = full_text
                current["character_count"] = len(full_text)
                records.append(current)

        current = None

    for page in extracted["pages"]:
        page_number = page["page_number"]

        lines = page["text"].replace(
            "\r\n", "\n"
        ).replace("\r", "\n").split("\n")

        for raw_line in lines:
            line = re.sub(
                r"[ \t]+", " ", raw_line
            ).strip()

            if not line:
                continue

            is_section = (
                SECTION_WORD_PATTERN.match(line)
                or SECTION_PATTERN.match(line)
            )

            # INTEGRATION FIX: "Section 2.1 Payment. The Company shall pay..." is a clause with a run-in
            # heading, not a bare heading line - treat it as text so its content is not dropped.
            if is_section and (len(line) > 100 or re.search(r"[a-z]{3,}\.\s+[A-Z]", line)):
                is_section = None

            if is_section:
                save_current()
                current_section = line
                section_count += 1

                records.append({
                    "record_id": f"SEC-{section_count:04d}",
                    "record_type": "section",
                    "section_heading": line,
                    "clause_number": None,
                    "text": line,
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_filename": extracted["metadata"]["filename"],
                    "extraction_method": page["extraction_method"],
                    "character_count": len(line)
                })
                continue

            match = CLAUSE_PATTERN.match(line)

            if match:
                save_current()
                clause_count += 1

                current = {
                    "record_id": f"CLAUSE-{clause_count:05d}",
                    "record_type": "clause",
                    "section_heading": current_section,
                    "clause_number": match.group(1),
                    "text_parts": [match.group(2)],
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_filename": extracted["metadata"]["filename"],
                    "extraction_method": page["extraction_method"]
                }

            elif current is not None:
                current["text_parts"].append(line)
                current["page_end"] = page_number

            else:
                clause_count += 1

                current = {
                    "record_id": f"BLOCK-{clause_count:05d}",
                    "record_type": "text_block",
                    "section_heading": current_section,
                    "clause_number": None,
                    "text_parts": [line],
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_filename": extracted["metadata"]["filename"],
                    "extraction_method": page["extraction_method"]
                }

    save_current()

    for record in records:
        record["source"] = {
            "filename": extracted["metadata"]["filename"],
            "sha256": extracted["metadata"]["sha256"],
            "page_start": record["page_start"],
            "page_end": record["page_end"]
        }

    return {
        "document_metadata": extracted["metadata"],
        "pages": extracted["pages"],
        "records": records,
        "warnings": extracted["warnings"],
        "processed_at_utc": datetime.now(timezone.utc).isoformat()
    }


def main():
    st.set_page_config(
        page_title="Document Hunter",
        page_icon="📄",
        layout="wide"
    )

    st.title("Document Hunter")
    st.subheader("PDF Processing and Clause Segmentation")

    st.write(
        "Upload a legal document to extract text, identify sections "
        "and clauses, preserve page numbers, and export source metadata."
    )

    uploaded_file = st.file_uploader(
        "Upload a legal PDF",
        type=["pdf"]
    )

    enable_ocr = st.checkbox(
        "Enable OCR for scanned PDFs",
        value=True
    )

    if enable_ocr:
        st.caption(
            "OCR requires Tesseract to be installed separately. "
            "Text extraction still works without OCR."
        )

    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            try:
                with st.spinner("Extracting and segmenting document..."):
                    pdf_bytes = uploaded_file.getvalue()

                    extracted = extract_pdf(
                        pdf_bytes,
                        uploaded_file.name,
                        enable_ocr
                    )

                    result = segment_document(extracted)

                    output_dir = Path("outputs")
                    output_dir.mkdir(exist_ok=True)

                    output_path = output_dir / (
                        Path(uploaded_file.name).stem
                        + "_processed.json"
                    )

                    json_data = json.dumps(
                        result,
                        indent=2,
                        ensure_ascii=False
                    )

                    output_path.write_text(
                        json_data,
                        encoding="utf-8"
                    )

                    st.session_state["result"] = result
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.session_state["json_data"] = json_data
                    st.session_state["output_path"] = str(output_path)

                st.success("Document processed successfully.")

            except Exception as error:
                st.error(f"Processing failed: {error}")


    if "result" in st.session_state:
        result = st.session_state["result"]
        metadata = result["document_metadata"]
        records = result["records"]

        sections = [
            r for r in records
            if r["record_type"] == "section"
        ]

        clauses = [
            r for r in records
            if r["record_type"] == "clause"
        ]

        blocks = [
            r for r in records
            if r["record_type"] == "text_block"
        ]

        st.divider()
        st.subheader("Processing Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("PDF Pages", metadata["page_count"])
        col2.metric("Sections", len(sections))
        col3.metric("Numbered Clauses", len(clauses))
        col4.metric("Other Text Blocks", len(blocks))

        st.subheader("Document Metadata")
        st.json(metadata)

        if result["warnings"]:
            st.warning("Some pages could not be OCR-processed.")

            for warning in result["warnings"]:
                st.write(warning)

        if not records:
            st.warning(
                "No text was extracted. The PDF may be blank, "
                "scanned, or unreadable."
            )

        st.subheader("Extracted Sections and Clauses")

        for record in records:
            label = (
                record.get("clause_number")
                or record.get("section_heading")
                or record["record_id"]
            )

            with st.expander(
                f'{record["record_id"]} | {label} | '
                f'Pages {record["page_start"]}-{record["page_end"]}'
            ):
                st.write(record["text"])

                st.caption(
                    f'Source: {record["source_filename"]} | '
                    f'Method: {record["extraction_method"]}'
                )

        st.download_button(
            label="Download Processed JSON",
            data=st.session_state["json_data"],
            file_name="processed_document.json",
            mime="application/json"
        )

        st.caption(
            f'JSON also saved locally to: '
            f'{st.session_state["output_path"]}'
        )


if __name__ == "__main__":
    main()
