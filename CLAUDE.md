# doc-assistant

Ask plain-English questions about any PDF and get answers with page citations. Fully local — no API keys, no cloud services.

## Vault context
For current project state, read:
`D:\Vaults\DevBrain\Projects\doc-assistant\Current-State.md`

## Stack
| Layer | Choice |
|-------|--------|
| PDF parsing | pymupdf |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | chromadb (local persistent) |
| LLM | qwen2.5:7b via Ollama |
| CLI | click |
| UI | gradio |

## Hard rules
- Keep fully local — no API keys, no cloud services, no external HTTP calls from core pipeline
- Chunks must remain page-aware; never let a chunk straddle two pages
- SHA-256 dedup must be preserved on index — re-indexing same file must be a no-op

## Last Session
<!-- Claude writes here at session end -->
