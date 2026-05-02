# doc-assistant

Ask plain-English questions about any PDF and get answers with **page citations**.

Built for local use — no API keys, no cloud services.

## Stack

| Component | Package | Why |
|---|---|---|
| PDF parsing | `pymupdf` | Page-accurate text extraction, handles dense legal formatting |
| Embeddings | `nomic-embed-text` via Ollama | 270MB, fast, no separate service |
| Vector DB | `chromadb` | Persistent local store, metadata filtering |
| LLM | `qwen2.5:7b` via Ollama | Best quality-per-GB at this size; beats Mistral 7B on reasoning/summarization |
| CLI | `click` | `index`, `ask`, `list`, `clear` |
| UI | `gradio` | Drag-drop upload + chat with citation display |

## Prerequisites

```powershell
# Pull models (one-time, ~5.3 GB total)
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### CLI

```powershell
# Index a PDF
python cli.py index docs\rental.pdf

# Ask a question (all indexed PDFs)
python cli.py ask "What is the security deposit amount?"

# Ask restricted to one file
python cli.py ask "What is the notice period?" --file rental.pdf

# List indexed files
python cli.py list

# Clear the vector store
python cli.py clear
```

### Gradio UI

```powershell
python app.py
# Open http://localhost:7860
```

Upload a PDF, then type questions in the chat box. Answers show `[filename, p.N]` citations.

## How it works

```
PDF → pages (PyMuPDF) → chunks (500 chars, 50 overlap, page-aware)
    → embed (nomic-embed-text) → ChromaDB (persisted)

Query → embed → top-4 similarity search → build prompt with citations
      → qwen2.5:7b → answer + source list
```

**Why page-aware chunking?** Chunks never straddle pages, so `[rental.pdf, p.4]` always points to the right page in a PDF viewer.

**Why SHA-256 dedup?** Re-running `index` on the same file is a no-op — no duplicate embeddings accumulate.

**Hallucination guard:** The prompt instructs the LLM to reply "I couldn't find this in the provided documents." if the retrieved context doesn't contain the answer.

## Project layout

```
doc-assistant/
├── src/
│   ├── config.py        # model names, chunk size, paths
│   ├── pdf_loader.py    # PyMuPDF extraction
│   ├── chunker.py       # page-aware splitting
│   ├── embeddings.py    # nomic-embed-text wrapper
│   ├── vector_store.py  # ChromaDB operations
│   ├── indexer.py       # orchestrates index pipeline + dedup
│   ├── retriever.py     # similarity search
│   ├── llm.py           # qwen2.5:7b wrapper
│   └── rag.py           # prompt + generation + citation formatting
├── cli.py               # click CLI
├── app.py               # gradio UI
├── docs/                # put your PDFs here (gitignored)
└── chroma_db/           # persisted vector store (gitignored)
```
