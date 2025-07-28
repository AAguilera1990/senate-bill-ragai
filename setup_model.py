# setup_model.py

from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import TokenTextSplitter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from langchain.llms import HuggingFacePipeline

# Constants
RAG_MODEL_ID = "mistralai/Mistral-7B-v0.1"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"

# Load quantized Mistral model as LLM
def load_llm():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16"
    )

    tokenizer = AutoTokenizer.from_pretrained(RAG_MODEL_ID, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        RAG_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        token=True
    )

    hf_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=200
    )
    return HuggingFacePipeline(pipeline=hf_pipe)


# Load Sentence-Transformer embedder
def get_embedder():
    return SentenceTransformerEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": "cuda"}
    )


# Return a safe token-based splitter under embedder limit
def get_text_splitter(chunk_size=200, chunk_overlap=50):
    return TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
