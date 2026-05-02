from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Chunk:
    text: str
    file: str       # basename of the PDF
    page: int       # 1-based
    chunk_idx: int  # within this page


def chunk(
    pages: list[tuple[int, str]],
    file_path: str | Path,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split each page's text into overlapping char-based chunks.
    Never merges across page boundaries so citations stay accurate.
    """
    filename = Path(file_path).name
    chunks: list[Chunk] = []

    for page_no, text in pages:
        idx = 0
        chunk_idx = 0
        while idx < len(text):
            end = idx + size
            snippet = text[idx:end]

            # Nudge end to next sentence boundary if mid-sentence
            if end < len(text):
                boundary = _find_sentence_end(text, end)
                snippet = text[idx:boundary]
                next_start = boundary - overlap
            else:
                next_start = len(text)

            snippet = snippet.strip()
            if snippet:
                chunks.append(Chunk(text=snippet, file=filename, page=page_no, chunk_idx=chunk_idx))
                chunk_idx += 1

            idx = max(next_start, idx + 1)  # always advance at least 1

    return chunks


def _find_sentence_end(text: str, near: int) -> int:
    """Return index just after the next sentence-ending punctuation near `near`."""
    for offset in range(0, min(100, len(text) - near)):
        ch = text[near + offset]
        if ch in ".!?\n":
            return near + offset + 1
    return near
