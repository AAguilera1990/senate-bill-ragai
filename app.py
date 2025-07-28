# app.py

import gradio as gr
from setup_model import model, tokenizer, chroma
from rag_chain import build_rag_chain, ask_question

# Step 1: Initialize the RAG chain
rag_chain = build_rag_chain(model, tokenizer, chroma)

# Step 2: Define Gradio interface function
def generate_answer(user_input):
    if not user_input.strip():
        return "Please enter a question."
    
    result = ask_question(rag_chain, tokenizer, user_input)
    if result is None:
        return "[Error] Question exceeded token limit."
    
    answer = result["result"].strip()

    # Format source chunks nicely
    sources = ""
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk", "?")
        sources += f"• {src} (chunk {chunk})\n"

    return f"### 📘 Answer:\n{answer}\n\n---\n### 🔍 Sources Used:\n{sources}"

# Step 3: Create Gradio UI
demo = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about the Senate bill..."),
    outputs=gr.Markdown(),
    title="🧠 Senate Bill RAG AI",
    description="Ask any question about the uploaded budget legislation PDF. Powered by Mistral 7B and MiniLM embeddings.",
    examples=[
    "What sections describe Medicare funding?",
    "How is small business support addressed?",
    "What are the core provisions of the bill?"
],
    theme="default"
)


# Step 4: Launch app (for local dev) or expose via Spaces
if __name__ == "__main__":
    demo.launch()  # For local dev; in Hugging Face Spaces this is auto-run
