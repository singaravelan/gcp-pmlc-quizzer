"""
Document parsing utilities for PDF, DOCX, and TXT files.

Imports for heavy libraries (pypdf, docx) are deferred to function bodies
to avoid slowing Streamlit's cold start.
"""
from __future__ import annotations

import io
from pathlib import Path

from config.settings import MAX_FILE_SIZE_MB


def parse_document(file_bytes: bytes, filename: str) -> str:
    """
    Parse an uploaded file's bytes into plain text.

    Args:
        file_bytes: Raw bytes from the uploaded file.
        filename: Original filename (used for extension detection).

    Returns:
        Extracted plain-text content.

    Raises:
        ValueError: Unsupported file extension.
        RuntimeError: Parsing failure.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(file_bytes)
    elif suffix == ".docx":
        return _parse_docx(file_bytes)
    elif suffix == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: PDF, DOCX, TXT."
        )


def _parse_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    if not pages:
        raise RuntimeError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image. Try a text-based PDF."
        )
    return "\n\n".join(pages)


def _parse_docx(file_bytes: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        raise RuntimeError("No text found in this DOCX file.")
    return "\n\n".join(paragraphs)


def validate_file_size(size_bytes: int, filename: str) -> str | None:
    """
    Validate file size against the configured maximum.

    Returns an error message string if invalid, None if OK.
    """
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            f"'{filename}' is {size_mb:.1f} MB, "
            f"which exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
    return None


def truncate_for_context(text: str, max_chars: int = 80_000) -> str:
    """
    Truncate text to fit within AI context limits.
    80,000 chars ≈ 20K tokens — well within claude-sonnet-4-6's 1M context.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... document truncated for context window ...]"
