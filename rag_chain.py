# rag_chain.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

def build_rag_chain(model, tokenizer, chroma: Chroma, max_new_tokens=500, k=10):
    """
    Builds and returns a RetrievalQA chain with a custom prompt and retriever.
    """
    # 1. Define your tree of thought prompt template
    prompt_template = """
You are a neutral legal assistant. Answer the user's question **only using the context provided below**, without any outside knowledge or opinions. Be concise, factual, and cite specific sections or language from the bill.

Context:
{context}

Question: {question}

Instructions:
- Do not guess. If the answer is not in the context, reply: "The bill does not provide sufficient information to answer this."
- When possible, include **quotes** or clear paraphrases from the text.
- Always mention the **section number** or **page number** if available.

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.strip()
    )

    # 2. Use pre-wrapped model passed from setup_model
    llm = model

    # 3. Compute token budget diagnostics
    static_example = prompt_template.format(context="", question="")
    static_tokens = len(tokenizer(static_example).input_ids)
    context_budget = 8000 - static_tokens - max_new_tokens - 64
    print(f"[RAG] Prompt static tokens: {static_tokens}")
    print(f"[RAG] Max context tokens available: {context_budget}")

    # 4. Create retriever from ChromaDB
    retriever = chroma.as_retriever(search_kwargs={"k": k})

    # 5. Create RAG chain with prompt
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return rag_chain


def ask_question(rag_chain, tokenizer, question: str, max_qtokens=300):
    """
    Run a single query through the RAG chain with validation.
    Returns the result object and prints sources.
    """
    qlen = len(tokenizer(question).input_ids)
    if qlen > max_qtokens:
        print(f"[Error] Question too long: {qlen} tokens. Limit is {max_qtokens}.")
        return None

    result = rag_chain(question)

    print("\n=== Answer ===")
    print(result["result"].strip())

    print("\n=== Source Chunks (k=10) ===")
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk", "?")
        print(f"• {src} (chunk {chunk})")

    return result


def start_cli_loop(rag_chain, tokenizer):
    """
    Launch a CLI loop for interactive questioning.
    """
    while True:
        user_q = input("\nEnter your question (blank to quit): ").strip()
        if not user_q:
            print("Goodbye!")
            break
        ask_question(rag_chain, tokenizer, user_q)
