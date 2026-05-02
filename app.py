import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from src import indexer, rag

# Track indexed files this session (display only — Chroma persists them)
_indexed: list[str] = []


def handle_upload(file_path) -> str:
    if file_path is None:
        return "No file uploaded."
    # Gradio 4.x passes a string path; older versions pass an object with .name
    path = file_path if isinstance(file_path, str) else file_path.name
    result = indexer.index_pdf(path)
    name = Path(path).name
    if result["skipped"]:
        return f"**{name}** was already indexed — ready to query."
    _indexed.append(name)
    return f"**{name}** indexed: {result['chunks']} chunks across {result['pages']} pages."


def handle_question(question: str, history: list, file_filter: str) -> tuple[list, str]:
    if not question.strip():
        return history, ""

    file = file_filter.strip() if file_filter.strip() else None
    result = rag.answer(question, file=file)

    answer_text = result["answer"]

    # Format source citations
    if result["sources"]:
        citations = "  \n".join(
            f"`[{s['file']}, p.{s['page']}]`" for s in result["sources"]
        )
        answer_text += f"\n\n**Sources:**  \n{citations}"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer_text})
    return history, ""


with gr.Blocks(title="Doc Assistant") as demo:
    gr.Markdown("# Doc Assistant\nAsk plain-English questions about your PDFs. Answers include page citations.")

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            index_btn = gr.Button("Index PDF", variant="secondary")
            upload_status = gr.Markdown("")
            file_filter = gr.Textbox(
                label="Restrict to file (optional)",
                placeholder="e.g. rental.pdf",
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=400)
            question_box = gr.Textbox(
                label="Your question",
                placeholder="What is the security deposit amount?",
            )
            ask_btn = gr.Button("Ask", variant="primary")

    index_btn.click(fn=handle_upload, inputs=upload, outputs=upload_status)
    ask_btn.click(
        fn=handle_question,
        inputs=[question_box, chatbot, file_filter],
        outputs=[chatbot, question_box],
    )
    question_box.submit(
        fn=handle_question,
        inputs=[question_box, chatbot, file_filter],
        outputs=[chatbot, question_box],
    )

if __name__ == "__main__":
    demo.launch()
