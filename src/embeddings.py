import ollama
from . import config


def embed(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text string."""
    vectors = []
    for text in texts:
        response = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
        vectors.append(response["embedding"])
    return vectors
