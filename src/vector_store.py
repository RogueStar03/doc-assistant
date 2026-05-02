from __future__ import annotations

import chromadb
from . import config
from .chunker import Chunk

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _client


def get_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(config.COLLECTION_NAME)


def get_meta_collection() -> chromadb.Collection:
    return _get_client().get_or_create_collection(config.META_COLLECTION)


def upsert(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    col = get_collection()
    ids = [f"{c.file}::p{c.page}::c{c.chunk_idx}" for c in chunks]
    metadatas = [{"file": c.file, "page": c.page} for c in chunks]
    col.upsert(ids=ids, embeddings=vectors, documents=[c.text for c in chunks], metadatas=metadatas)


def query(
    vector: list[float],
    k: int = config.TOP_K,
    where: dict | None = None,
) -> list[dict]:
    """
    Returns a list of result dicts with keys: text, file, page, distance.
    """
    col = get_collection()
    kwargs: dict = {"query_embeddings": [vector], "n_results": k, "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)
    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"text": doc, "file": meta["file"], "page": meta["page"], "distance": dist})
    return out


def list_indexed_files() -> list[str]:
    col = get_meta_collection()
    result = col.get(include=["documents"])
    return result["documents"]


def add_file_hash(filename: str, sha256: str) -> None:
    col = get_meta_collection()
    col.upsert(ids=[filename], documents=[sha256])


def get_file_hash(filename: str) -> str | None:
    col = get_meta_collection()
    result = col.get(ids=[filename], include=["documents"])
    docs = result.get("documents", [])
    return docs[0] if docs else None


def clear_all() -> None:
    client = _get_client()
    for name in [config.COLLECTION_NAME, config.META_COLLECTION]:
        try:
            client.delete_collection(name)
        except Exception:
            pass
