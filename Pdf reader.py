"""Read text from an uploaded PDF file."""

from pypdf import PdfReader


class PDFError(Exception):
    """Raised when a PDF cannot be read. The message is safe to show to users."""


def extract_text_from_pdf(uploaded_file):
    """Return (text, number_of_pages) from a Streamlit uploaded file."""
    # 1. Try to open the PDF
    try:
        reader = PdfReader(uploaded_file)
    except Exception:
        raise PDFError(
            "This file could not be opened. It may be corrupted or not a real "
            "PDF. Please try another file."
        )

    # 2. Handle password-protected PDFs
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise PDFError(
                "This PDF is password-protected. Please remove the password "
                "and upload it again."
            )

    # 3. Extract text page by page
    pages_text = []
    try:
        total_pages = len(reader.pages)
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""  # skip a broken page instead of failing completely
            if text.strip():
                pages_text.append(text.strip())
    except Exception:
        raise PDFError("Something went wrong while reading the pages of this PDF.")

    full_text = "\n\n".join(pages_text)

    # 4. Handle empty / scanned documents
    if len(full_text.strip()) < 30:
        raise PDFError(
            "No readable text was found. This PDF may contain only scanned "
            "images or be empty. Please upload a PDF with selectable text."
        )

    return full_text, total_pages
