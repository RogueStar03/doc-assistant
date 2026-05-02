# How doc-assistant Works — A Plain-English Deep Dive

---

## The problem we are solving

You have a 50-page rental agreement. You want to know: "What is the security deposit?"

You cannot paste the whole PDF into ChatGPT — it is too long. Even if you could, the model
might make things up. You want the answer AND the exact page it came from.

**The solution is RAG — Retrieval-Augmented Generation.**

Instead of feeding the whole document to the AI, we:
1. Break the document into small pieces
2. Store those pieces in a special database
3. When you ask a question, find the 4 most relevant pieces
4. Feed ONLY those 4 pieces to the AI and ask it to answer

The AI reads 4 short paragraphs instead of 50 pages. It is faster, more accurate,
and we can tell it to cite which page each piece came from.

---

## The two phases

```
PHASE 1 — INDEXING (do once per PDF)
PDF → read pages → split into chunks → convert to numbers → store in database

PHASE 2 — QUERYING (every time you ask a question)
Question → convert to numbers → find similar chunks → ask AI → get answer with citations
```

Why "convert to numbers"? Because computers cannot compare meaning directly.
A number-vector called an "embedding" represents the meaning of text.
Similar meaning = similar numbers. This is explained fully below.

---

## Step 1 — Reading the PDF (`src/pdf_loader.py`)

**Goal:** Get the text out of the PDF, keeping track of which page each sentence is on.

**Library:** `pymupdf` (imported as `fitz`) — a fast PDF reader.

**What we need:**
We do NOT want one giant string of all the text. We need to know "this sentence is on page 4"
so that later we can tell the user "the answer is on page 4."

**How the code works:**

```python
import fitz

doc = fitz.open("rental.pdf")      # open the PDF

for i, page in enumerate(doc, start=1):   # loop through pages, counting from 1
    text = page.get_text("text")           # extract the text on this page as a string
```

`enumerate(doc, start=1)` means: go through each page, and give me a counter
starting at 1 (not 0). So first page → i=1, second page → i=2, etc.

**Example — what goes in vs what comes out:**

```
INPUT:  rental.pdf  (a 10-page PDF)

OUTPUT (what the function returns):
[
  (1, "RENTAL AGREEMENT\nThis agreement is made between..."),
  (2, "1. TERM\nThe tenancy shall commence on..."),
  (3, "2. RENT\nThe monthly rent shall be $1,200..."),
  (4, "3. SECURITY DEPOSIT\nTenant shall pay $2,400 as deposit..."),
  ...
]
```

Each item in the list is a tuple: (page_number, text_on_that_page).

---

## Step 2 — Splitting into chunks (`src/chunker.py`)

**Goal:** Break each page's text into smaller overlapping pieces.

**Why not just use the full page?**
A page of a legal document can be 2,000 characters. When we do the similarity search later,
we want to find the ONE paragraph that answers the question — not a whole page full of
unrelated clauses. Smaller pieces = more precise retrieval.

**Why overlap?**
Imagine a sentence is split exactly at the chunk boundary:
```
Chunk 1: "...the tenant must give 30 days written"
Chunk 2: "notice before vacating the property..."
```
The full sentence "30 days written notice" only appears completely in neither chunk.
With overlap, the end of chunk 1 is repeated at the start of chunk 2, so the full
sentence survives in at least one chunk.

**What is a Chunk?**
We defined a small data container called `Chunk` using Python's `dataclass`:

```python
@dataclass
class Chunk:
    text: str       # the actual text
    file: str       # which PDF file (e.g. "rental.pdf")
    page: int       # which page (e.g. 4)
    chunk_idx: int  # which chunk on that page (0, 1, 2, ...)
```

A `dataclass` is just a convenient way to bundle related values together,
like a named tuple, but cleaner.

**The key rule: NEVER merge across pages.**
If page 4 ends mid-sentence and page 5 continues it, we do NOT merge them.
Why? Because we need to be able to say "this answer is from page 4."
If the chunk mixes page 4 and page 5, that citation would be a lie.

**Example — what goes in vs what comes out:**

```
INPUT:
  page 4 text = "The security deposit shall be $2,400. This amount is held in
  trust by the landlord. It will be returned within 30 days of vacating..."

OUTPUT (with CHUNK_SIZE=60 for illustration):
  Chunk(text="The security deposit shall be $2,400.",        file="rental.pdf", page=4, chunk_idx=0)
  Chunk(text="$2,400. This amount is held in trust by...",   file="rental.pdf", page=4, chunk_idx=1)
  Chunk(text="held in trust by the landlord. It will be...", file="rental.pdf", page=4, chunk_idx=2)
```

Notice: "2,400." appears at both the end of chunk 0 and the start of chunk 1 — that is the overlap.
In our real code, CHUNK_SIZE=500 and CHUNK_OVERLAP=50 (characters).

---

## Step 3 — Converting text to numbers ("embeddings") (`src/embeddings.py`)

**This is the core idea of modern AI search. Read this carefully.**

**Goal:** Convert a piece of text into a list of numbers (called a vector or embedding)
that represents its *meaning*.

**Why numbers?**
Computers can calculate whether two lists of numbers are similar — it is just math.
They cannot directly compare whether two sentences have similar meaning.
Embeddings are how we bridge that gap.

**The key insight:**
Text with similar meaning gets converted to similar numbers.

```
embed("What is the security deposit?")
→ [0.12, -0.45, 0.88, 0.23, ...]   (768 numbers)

embed("The tenant must pay $2,400 as deposit.")
→ [0.11, -0.43, 0.85, 0.25, ...]   (768 numbers, very similar!)

embed("The cat sat on the mat.")
→ [-0.78, 0.12, -0.34, 0.91, ...]  (768 numbers, very different)
```

The 768 numbers define a position in a 768-dimensional "meaning space."
Sentences about deposits end up near each other in that space.
Sentences about cats end up far away.

**Library:** `ollama` — Python package that talks to the Ollama program running on your machine.
Ollama runs the `nomic-embed-text` model, which is a small AI specifically trained
to convert text to embeddings. It has no language abilities — it only does this conversion.

**How the code works:**

```python
import ollama

response = ollama.embeddings(model="nomic-embed-text", prompt="What is the deposit?")
vector = response["embedding"]   # a list of 768 floats
```

**What goes in vs what comes out:**

```
INPUT:  ["The deposit is $2,400.", "Notice period is 30 days."]

OUTPUT: [
    [0.12, -0.45, 0.88, ...],   # 768 numbers for first sentence
    [-0.23, 0.67, 0.11, ...],   # 768 numbers for second sentence
]
```

---

## Step 4 — Storing in the vector database (`src/vector_store.py`)

**Goal:** Store all the chunk-vectors so we can quickly find the ones most similar
to a question-vector later.

**Library:** `chromadb` — a vector database that runs locally on your machine.
A vector database is like a regular database (store and retrieve data) but it
has one superpower: "find me the N items whose vectors are most similar to this vector."
This is called a similarity search or nearest-neighbour search.

**Regular database vs vector database:**
```
Regular (SQL):  "Find rows WHERE deposit = 2400"   → exact match
Vector DB:      "Find chunks MOST SIMILAR TO this question-vector"  → meaning match
```

**How ChromaDB is set up:**

We use `PersistentClient` so the data is saved to disk in `./chroma_db/`. 
If you restart the app, the data is still there. No re-indexing needed.

We create two collections (like tables):
- `documents` — stores all the chunks (text + vector + page number + filename)
- `indexed-files` — stores which files have been indexed (filename → SHA-256 hash)

**`upsert` = insert or update**
If the chunk already exists (same ID), update it. If not, insert it.
The ID for each chunk is: `"rental.pdf::p4::c0"` (file, page, chunk index).
This means re-indexing the same file is safe — it just overwrites, no duplicates.

**How similarity search works:**

```
We have stored 200 chunk-vectors.

Question: "What is the security deposit?"
  → embed it → [0.12, -0.45, 0.88, ...]

ChromaDB calculates the "distance" between this vector
and all 200 stored vectors. Smaller distance = more similar.

Returns the 4 chunks with the smallest distance:
  1. "The security deposit shall be $2,400..." (page 4) — distance: 0.05
  2. "Deposit is refunded within 30 days..."   (page 7) — distance: 0.12
  3. "Landlord holds deposit in trust..."      (page 4) — distance: 0.18
  4. "Tenant may not use deposit as rent..."   (page 5) — distance: 0.31
```

---

## Step 5 — Deduplication in the indexer (`src/indexer.py`)

**Goal:** Don't re-index a PDF that was already indexed.

**Problem:** If you run `python cli.py index rental.pdf` twice, we'd embed everything twice,
creating duplicate chunks. The similarity search would return duplicate results.

**Solution:** SHA-256 file hash.

SHA-256 is a function that takes any file and produces a unique 64-character string.
Same file = same string. Even one character changed = completely different string.

```python
import hashlib

data = open("rental.pdf", "rb").read()   # read file as raw bytes
hash = hashlib.sha256(data).hexdigest()
# → "a3f2d9e1b8c4..."  (always the same for this exact file)
```

When you run `index rental.pdf`:
1. Calculate its SHA-256 hash
2. Check if that hash is stored in the `indexed-files` collection
3. If yes → print "already indexed, skipping"
4. If no → run the full pipeline, then store the hash

**The full indexer flow:**

```
index_pdf("rental.pdf")
  ├── hash = sha256("rental.pdf")           → "a3f2d9..."
  ├── get_file_hash("rental.pdf")           → None (not seen before)
  ├── pdf_loader.load("rental.pdf")         → [(1, "text..."), (2, "text..."), ...]
  ├── chunker.chunk(pages, "rental.pdf")    → [Chunk(...), Chunk(...), ...]
  ├── embeddings.embed([chunk.text, ...])   → [[0.12, ...], [-0.45, ...], ...]
  ├── vector_store.upsert(chunks, vectors)  → saved to chroma_db/
  └── add_file_hash("rental.pdf", "a3f2d9") → saved to indexed-files collection
```

---

## Step 6 — Retrieving relevant chunks (`src/retriever.py`)

**Goal:** Given a question, find the 4 most relevant chunks.

This is just one function that calls two things we already built:

```python
def retrieve(question, k=4, file=None):
    vec = embeddings.embed([question])[0]      # embed the question
    return vector_store.query(vec, k=k)        # find top-4 similar chunks
```

The optional `file` parameter adds a filter: only search within one PDF.
Useful if you have 3 PDFs indexed and only want to search the rental agreement.

**Example:**

```
INPUT:  question = "What is the security deposit?"

Step 1: embed → [0.12, -0.45, 0.88, ...]
Step 2: ChromaDB finds top 4 matches

OUTPUT:
[
  {"text": "The security deposit shall be $2,400...", "file": "rental.pdf", "page": 4, "distance": 0.05},
  {"text": "Deposit returned within 30 days...",       "file": "rental.pdf", "page": 7, "distance": 0.12},
  {"text": "Landlord holds deposit in trust...",       "file": "rental.pdf", "page": 4, "distance": 0.18},
  {"text": "Tenant may not use deposit as rent...",   "file": "rental.pdf", "page": 5, "distance": 0.31},
]
```

---

## Step 7 — Generating the answer (`src/llm.py` + `src/rag.py`)

**`llm.py` — talking to qwen2.5:7b**

This is just a wrapper around `ollama.chat`:

```python
import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": prompt}]
)
answer = response["message"]["content"]
```

`ollama.chat` works like a one-shot conversation with the LLM.
We give it one message (the user prompt), it gives back one response.
There is no memory between calls — each call is independent.

**`rag.py` — building the prompt and returning citations**

This is the most important piece. The quality of the answer depends entirely
on how well the prompt is written.

**What is a prompt?**
Everything we send to the LLM is called a prompt. We include:
1. Instructions (what role to play, what rules to follow)
2. Context (the retrieved chunks)
3. The actual question

The LLM cannot distinguish between "instructions" and "context" at a code level —
it all arrives as one text string. That is why prompt engineering matters.

**The prompt we build:**

```
You are a careful assistant answering questions from the user's documents.
Use ONLY the context below. If the answer is not in the context, reply exactly:
"I couldn't find this in the provided documents."
Cite each fact you use as [filename, p.N]. Multiple citations are fine.

Context:
[1] (rental.pdf, p.4)
The security deposit shall be $2,400. This amount is held in trust...

[2] (rental.pdf, p.7)
The deposit will be returned within 30 days of vacating...

[3] (rental.pdf, p.4)
Landlord holds deposit in trust for the duration of...

[4] (rental.pdf, p.5)
Tenant may not use the deposit as the last month's rent...

Question: What is the security deposit?
Answer:
```

**Why "Use ONLY the context"?**
Without this instruction, the LLM will use its training knowledge to fill gaps.
If the document says "$2,400" but the LLM was trained on similar agreements
that usually say "$1,000", it might blend the two. We want it to only read what we gave it.

**Why the "I couldn't find this" fallback?**
If you ask "What is the wifi password?" and that is not in a rental agreement,
without this instruction the LLM might invent one. With it, it will correctly say
"I couldn't find this in the provided documents."

**The full rag.py flow:**

```
answer("What is the security deposit?")
  ├── retriever.retrieve(question)   → 4 chunks (from step 6)
  ├── build context string           → "[1] (rental.pdf, p.4)\n..."
  ├── assemble full prompt           → instructions + context + question
  ├── llm.generate(prompt)           → "The security deposit is $2,400 [rental.pdf, p.4]..."
  └── return {
        "answer": "The security deposit is $2,400 [rental.pdf, p.4]...",
        "sources": [{"file": "rental.pdf", "page": 4, "text": "..."}, ...]
      }
```

---

## Step 8 — The CLI (`cli.py`)

**Library:** `click` — makes it easy to build command-line tools.
Without click, you'd have to parse `sys.argv` manually. With click, you just
decorate a function and it handles argument parsing, help text, and errors.

```python
@click.group()          # this function is the root command group
def cli():
    pass

@cli.command()          # this is a subcommand of cli
@click.argument("path") # it takes one positional argument called "path"
def index(path):
    ...
```

Running `python cli.py index rental.pdf` → click calls `index(path="rental.pdf")`.

The `--confirmation_option` on `clear` is click's built-in "are you sure?" prompt.
It stops accidental data deletion.

---

## Step 9 — The Gradio UI (`app.py`)

**Library:** `gradio` — turns Python functions into web UIs in ~10 lines.

The UI has two columns:
```
LEFT COLUMN                    RIGHT COLUMN
┌─────────────────┐           ┌──────────────────────────┐
│ Upload PDF      │           │ Chat history             │
│ [Index PDF btn] │           │                          │
│ Status message  │           │                          │
│                 │           │ Q: What is the deposit?  │
│ Restrict to     │           │ A: $2,400 [p.4]          │
│ file: _______   │           │                          │
│                 │           │ Your question: _______   │
│                 │           │ [Ask]                    │
└─────────────────┘           └──────────────────────────┘
```

**`gr.Blocks`** — lets us arrange components in a custom layout (as opposed to
`gr.Interface` which is a fixed single-function layout).

**How events are wired:**

```python
index_btn.click(fn=handle_upload, inputs=upload, outputs=upload_status)
```
Translation: when the button is clicked, call `handle_upload(upload.value)`,
and put whatever it returns into `upload_status`.

**`handle_upload(file_path)`**
- `file_path` is a string (the path to the temp file Gradio saved the upload to)
- We call `indexer.index_pdf(file_path)`
- Return a status message string → displayed under the button

**`handle_question(question, history, file_filter)`**
- Calls `rag.answer(question)` → gets answer + sources
- Appends to `history` as `{"role": "user", "content": "..."}` and `{"role": "assistant", "content": "..."}`
- Gradio 6 uses this dict format for chat messages
- Returns updated history + empty string (to clear the question box)

---

## Complete data flow — one full example

```
You drop "rental.pdf" into the UI and click "Index PDF"
│
├── handle_upload("C:/tmp/gradio_xyz/rental.pdf")
│     └── indexer.index_pdf(path)
│           ├── sha256(file) → "a3f2..."
│           ├── check indexed-files → not found
│           ├── pdf_loader.load() → [(1, "text"), (2, "text"), ..., (10, "text")]
│           ├── chunker.chunk()  → [Chunk(p1,c0), Chunk(p1,c1), ..., Chunk(p10,c3)]
│           │                        total: ~47 chunks
│           ├── embeddings.embed(47 texts) → 47 vectors of 768 numbers each
│           │   (calls nomic-embed-text 47 times via Ollama)
│           ├── vector_store.upsert(47 chunks, 47 vectors) → saved to chroma_db/
│           └── add_file_hash("rental.pdf", "a3f2...")
│
└── Status shows: "rental.pdf indexed: 47 chunks across 10 pages"

You type: "What is the security deposit?" and click Ask
│
├── handle_question("What is the security deposit?", [], "")
│     └── rag.answer("What is the security deposit?")
│           ├── retriever.retrieve("What is the security deposit?")
│           │     ├── embed question → [0.12, -0.45, 0.88, ...]
│           │     └── chroma query → top 4 chunks (p.4, p.7, p.4, p.5)
│           ├── build prompt with 4 chunks as context
│           └── llm.generate(prompt)
│               → "The security deposit is $2,400. This must be paid
│                  before the lease begins [rental.pdf, p.4]. It will
│                  be returned within 30 days [rental.pdf, p.7]."
│
└── Chat shows:
      You: What is the security deposit?
      Assistant: The security deposit is $2,400...
                 Sources: `[rental.pdf, p.4]`  `[rental.pdf, p.7]`
```

---

## Glossary

| Term | Plain English |
|---|---|
| **RAG** | Retrieval-Augmented Generation — find relevant pieces first, then generate answer from those pieces |
| **Embedding / Vector** | A list of numbers representing the *meaning* of text |
| **Chunk** | A small piece of a document (500 chars in our case) |
| **Similarity search** | Find stored vectors closest (most similar in meaning) to a query vector |
| **ChromaDB** | A database that stores vectors and can do similarity search |
| **Ollama** | A program that runs AI models locally on your machine |
| **nomic-embed-text** | A small AI model that only does one thing: text → embedding vector |
| **qwen2.5:7b** | A general-purpose LLM (7 billion parameters) that reads context and answers questions |
| **PersistentClient** | ChromaDB mode where data is saved to disk, not lost when app restarts |
| **SHA-256** | A fingerprint function for files — same file always gives same output |
| **Prompt** | Everything you send to an LLM (instructions + context + question all in one string) |
| **click** | Python library for building CLI tools with commands and arguments |
| **gradio** | Python library for building browser UIs by wrapping Python functions |
