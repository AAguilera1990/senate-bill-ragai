# rag_chain.py

from transformers import pipeline, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

def build_rag_chain(model, tokenizer, chroma: Chroma, max_new_tokens=500, k=10):
    """
    Builds and returns a RetrievalQA chain with a custom prompt and retriever.
    """
    # 1. Define your zero-shot prompt template
    prompt_template = """
    You are a legal assistant. Use the following context from the Bill guideline to answer clearly, citing section.
    Context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.strip()
    )

    # 2. Create the LLM pipeline (limit generation tokens)
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=max_new_tokens
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

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
