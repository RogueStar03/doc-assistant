from __future__ import annotations

from . import embeddings, vector_store, config


def retrieve(question: str, k: int = config.TOP_K, file: str | None = None) -> list[dict]:
    """
    Embed the question and return the top-k matching chunks.
    Optionally filter by filename.
    """
    vec = embeddings.embed([question])[0]
    where = {"file": file} if file else None
    return vector_store.query(vec, k=k, where=where)
