from __future__ import annotations

from . import retriever, llm, config

_SYSTEM = """\
You are a careful assistant answering questions from the user's documents.
Use ONLY the context below. If the answer is not in the context, reply exactly:
"I couldn't find this in the provided documents."
Cite each fact you use as [filename, p.N]. Multiple citations are fine.\
"""


def answer(question: str, file: str | None = None, k: int = config.TOP_K) -> dict:
    """
    Retrieve relevant chunks, build a cited prompt, and generate an answer.
    Returns {"answer": str, "sources": list[dict]}.
    Each source has keys: file, page, text.
    """
    chunks = retriever.retrieve(question, k=k, file=file)

    if not chunks:
        return {
            "answer": "I couldn't find this in the provided documents.",
            "sources": [],
        }

    context_lines = []
    for i, c in enumerate(chunks, start=1):
        context_lines.append(f"[{i}] ({c['file']}, p.{c['page']})\n{c['text']}")
    context = "\n\n".join(context_lines)

    prompt = f"{_SYSTEM}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    raw = llm.generate(prompt)

    return {
        "answer": raw,
        "sources": [{"file": c["file"], "page": c["page"], "text": c["text"]} for c in chunks],
    }
