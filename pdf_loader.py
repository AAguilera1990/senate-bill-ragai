# pdf_loader.py

import os
import re
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document


def simple_clean(text: str) -> str:
    text = text.replace('\f', ' ')
    text = re.sub(r'-\s*\n\s*', '', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_and_clean_pdfs(pdf_dir: str) -> List[Document]:
    """
    Loads all PDFs in a directory, cleans the text of each page, and returns:
    - A list of cleaned LangChain Document objects (1 per page)
    """
    clean_docs = []

    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                cleaned = simple_clean(doc.page_content)
                clean_doc = Document(page_content=cleaned, metadata={"source": fname})
                clean_docs.append(clean_doc)

    print(f"[INFO] Loaded and cleaned {len(clean_docs)} pages from {pdf_dir}")
    return clean_docs
