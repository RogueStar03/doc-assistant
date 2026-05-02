from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

CHROMA_PATH = str(BASE_DIR / "chroma_db")
DOCS_DIR = BASE_DIR / "docs"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

CHUNK_SIZE = 500      # characters
CHUNK_OVERLAP = 50    # characters
TOP_K = 4             # chunks retrieved per query

COLLECTION_NAME = "documents"
META_COLLECTION = "indexed-files"
