from typing import List
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter

def chunk_documents(
    documents: List[Document],
    cleaned_texts: List[str],
    chunk_size: int = 200,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Token-split each cleaned document using LangChain's TokenTextSplitter.
    Preserves metadata, adds 'chunk' index to metadata.
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = []
    for original_doc, cleaned_text in zip(documents, cleaned_texts):
        splits = splitter.split_text(cleaned_text)
        for idx, chunk_text in enumerate(splits):
            meta = original_doc.metadata.copy()
            meta["chunk"] = idx
            chunked_docs.append(Document(page_content=chunk_text, metadata=meta))
    
    print(f"[INFO] Chunked into {len(chunked_docs)} total chunks.")
    return chunked_docs


def embed_chunks_to_chroma(chunked_docs: List[Document], chroma):
    """
    Embed documents into ChromaDB and persist with metadata.
    """
    texts = [doc.page_content for doc in chunked_docs]
    metadatas = [doc.metadata for doc in chunked_docs]

    chroma.add_texts(texts=texts, metadatas=metadatas)
    print("[INFO] Embedded all chunks into ChromaDB.")


def preview_embeddings(chroma, limit: int = 3):
    """
    Show a preview of text snippets and embedding vector samples.
    """
    collection = chroma._collection
    result = collection.get(limit=limit, include=["documents", "embeddings"])

    for idx, (text, emb) in enumerate(zip(result["documents"], result["embeddings"]), start=1):
        print(f"--- Sample {idx} ---")
        snippet = text.replace("\n", " ")[:200]
        print("Text snippet:", snippet + ("..." if len(text) > 200 else ""))
        print("Embedding sample values:", emb[:5], "...")
        print("Embedding dimension:", len(emb))
        print()
