from __future__ import annotations

from pathlib import Path
import fitz  # pymupdf


def load(path: str | Path) -> list[tuple[int, str]]:
    """
    Extract text per page from a PDF.
    Returns a list of (1-based page number, page text) tuples.
    Skips pages with no extractable text.
    """
    path = Path(path)
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append((i, text))
    doc.close()
    return pages
