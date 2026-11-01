"""PDF bytes or path -> per-page text. Text PDFs only (no OCR)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pdfplumber

MAX_PAGES = 30
MIN_TOTAL_CHARS = 80  # below this, treat as scanned/empty


class ExtractError(ValueError):
    """User-safe message: shown directly in the UI."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, matches what the reader sees
    text: str


@dataclass
class Document:
    name: str
    pages: list[Page] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    def page_texts(self) -> dict[int, str]:
        return {p.number: p.text for p in self.pages}


Source = str | Path | bytes | BinaryIO


def _open(source: Source) -> pdfplumber.PDF:
    if isinstance(source, bytes):
        return pdfplumber.open(io.BytesIO(source))
    if isinstance(source, (str, Path)):
        return pdfplumber.open(source)
    return pdfplumber.open(source)  # file-like, e.g. Streamlit UploadedFile


def extract_pdf(source: Source, name: str = "document.pdf") -> Document:
    """Extract text from every page. Raises ExtractError with a user-facing message."""
    try:
        pdf = _open(source)
    except Exception as exc:
        raise ExtractError("This file could not be opened as a PDF.") from exc

    with pdf:
        if len(pdf.pages) == 0:
            raise ExtractError("This PDF has no pages.")
        if len(pdf.pages) > MAX_PAGES:
            raise ExtractError(f"This PDF has {len(pdf.pages)} pages. The limit is {MAX_PAGES}.")

        doc = Document(name=name)
        for index, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            text = "\n".join(line.rstrip() for line in raw.splitlines()).strip()
            doc.pages.append(Page(number=index, text=text))

    total_chars = sum(len(p.text) for p in doc.pages)
    if total_chars < MIN_TOTAL_CHARS:
        raise ExtractError(
            "This looks like a scanned PDF. Please upload a text-based PDF "
            "(one where you can select and copy the text)."
        )
    return doc


if __name__ == "__main__":
    import sys

    doc = extract_pdf(Path(sys.argv[1]))
    for page in doc.pages:
        print(f"--- page {page.number} ---")
        print(page.text)