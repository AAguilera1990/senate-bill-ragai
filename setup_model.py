# Module 1: Load Libraries & Initialize Models for RAG Pipeline (with correct embedder max length)

import os
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import TokenTextSplitter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

# # 1. Quantization config for Mistral 7B
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype="float16"
# )

RAG_MODEL_ID = "microsoft/phi-2"
HF_TOKEN = None
# HF_TOKEN = os.getenv("HF_TOKEN")

# # 2. Load tokenizer & quantized model for RAG generation
tokenizer = AutoTokenizer.from_pretrained(RAG_MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     RAG_MODEL_ID,
#     quantization_config=quant_config,
#     device_map="auto"
# )

model = AutoModelForCausalLM.from_pretrained(
    RAG_MODEL_ID,
    # token=HF_TOKEN
)


# 3. Wrap in a Hugging Face text-generation pipeline (limit new tokens)
hf_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=200
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 4. Initialize the Sentence‑Transformer embedder
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": "cpu"}
)

# 5. Determine the embedder’s true max token window via sentence-transformers
st = SentenceTransformer(EMBED_MODEL_ID)
embedder_max = st.max_seq_length  # typically 256

# 6. Configure a token‑based splitter safely under that limit (e.g., 200 tokens)
chunker = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=200,   # safely below embedder_max
    chunk_overlap=50  # context overlap
)

# 7. Initialize ChromaDB for vector storage & retrieval
chroma = Chroma(
    embedding_function=embeddings
)

model = llm  # Expose HuggingFacePipeline so app.py can import it
