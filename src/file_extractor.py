"""
file_extractor.py

This module provides functions to extract text from various file types,
including TXT, Markdown, PDFs, DOCX, source code files, and Google files.

For Google files, use the following file path formats:
  - Google Docs: "gdoc:<GOOGLE_DOC_ID>"
  - Google Sheets: "gsheet:<GOOGLE_SHEET_ID>"

The module delegates extraction of Google Docs and Sheets to the corresponding
functions in the google_integration module.
"""

import os
import pdfplumber
import chardet


def extract_text(file_path):
    """
    Extract text from various file types.

    Supported formats: TXT, MD, Python/JS/Lua source, PDF, DOCX,
    Google Docs, and Google Sheets.

    For Google Docs, use the file path format: "gdoc:<GOOGLE_DOC_ID>"
    For Google Sheets, use the file path format: "gsheet:<GOOGLE_SHEET_ID>"

    :param file_path: Path to the file or a special identifier for Google files.
    :return: Extracted text or None if unsupported.
    """
    # Check for Google Docs file format.
    if file_path.startswith("gdoc:"):
        doc_id = file_path[len("gdoc:") :]
        from src.google_integration import read_google_doc

        return read_google_doc(doc_id)

    # Check for Google Sheets file format.
    if file_path.startswith("gsheet:"):
        sheet_id = file_path[len("gsheet:") :]
        from src.google_integration import read_google_sheet

        return read_google_sheet(sheet_id)

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".md", ".py", ".js", ".lua"]:
        # Read text and source code files.
        with open(file_path, "rb") as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)["encoding"] or "utf-8"
        return raw_data.decode(encoding, errors="ignore")

    elif ext == ".pdf":
        # Read PDFs.
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])

    elif ext == ".docx":
        # Read Word documents.
        import docx

        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return None  # Unsupported file type
