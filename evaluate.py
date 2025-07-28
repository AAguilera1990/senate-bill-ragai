# ─── Add LLM‑as‑Judge with Detailed Rubric & Proper Scoring ───

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline_fn
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate

# 1. Load & wrap judge model (Phi 1.5B)
JUDGE_MODEL_ID = "microsoft/phi-1_5"
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, use_auth_token=True)
judge_model     = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_ID, trust_remote_code=True, device_map="auto", use_auth_token=True
)
judge_pipeline = hf_pipeline_fn(
    task="text-generation",
    model=judge_model,
    tokenizer=judge_tokenizer,
    return_full_text=False,
    max_new_tokens=200
)
judge_llm = HuggingFacePipeline(pipeline=judge_pipeline)

# 2. Define rubric text
rubric_text = """
Rubric for evaluation:
Faithfulness:
  1: Major hallucinations or contradictions.
  2: Significant inaccuracies; some correct.
  3: Partially faithful; most key points correct.
  4: Mostly faithful; minor omissions.
  5: Fully faithful and accurate.

Relevance:
  1: Unrelated to question/context.
  2: Minimally related.
  3: Somewhat relevant; misses aspects.
  4: Mostly relevant; minor gaps.
  5: Highly relevant; fully addresses question.
"""

# 3. Build judge prompt including rubric
judge_prompt = PromptTemplate(
    input_variables=["question", "answer", "context"],
    template=rubric_text + """
Question:
{question}

Answer:
{answer}

Context:
{context}

Provide:
1) Faithfulness (1–5):
2) Relevance    (1–5):
Justification:
"""
)
judge_chain = LLMChain(llm=judge_llm, prompt=judge_prompt)

# 4. Interactive RAG + Judge loop (assumes rag_chain & qtok defined)
while True:
    user_q = input("\nEnter question (blank to quit): ").strip()
    if not user_q:
        break

    # Enforce ≤300 tokens
    if len(qtok(user_q).input_ids) > 300:
        print("[Error] Question too long.")
        continue

    # Run RAG chain
    rag_res = rag_chain(user_q)
    answer  = rag_res["result"].strip()
    docs    = rag_res["source_documents"]

    print("\n=== RAG Answer ===\n", answer)

    # Prepare context
    ctx = "\n\n".join(d.page_content for d in docs[:3])

    # Display rubric once
    print(rubric_text)

    # Run judge evaluation
    eval_out = judge_chain.run({
        "question": user_q,
        "answer":   answer,
        "context":  ctx
    })

    print("\n=== Judge Evaluation ===\n", eval_out)
    print("\n" + "="*60)
