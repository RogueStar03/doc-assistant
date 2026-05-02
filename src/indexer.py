from __future__ import annotations

import hashlib
from pathlib import Path

from . import pdf_loader, chunker, embeddings, vector_store


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def index_pdf(path: str | Path) -> dict:
    """
    Index a PDF into the vector store.
    Returns a dict with keys: skipped (bool), chunks (int), pages (int).
    Skips if the same file (by SHA-256) was already indexed.
    """
    path = Path(path)
    filename = path.name
    sha = _sha256(path)

    existing = vector_store.get_file_hash(filename)
    if existing == sha:
        return {"skipped": True, "chunks": 0, "pages": 0}

    pages = pdf_loader.load(path)
    chunks = chunker.chunk(pages, path)
    vectors = embeddings.embed([c.text for c in chunks])
    vector_store.upsert(chunks, vectors)
    vector_store.add_file_hash(filename, sha)

    return {"skipped": False, "chunks": len(chunks), "pages": len(pages)}
