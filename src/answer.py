import time
import sys
from pathlib import Path

import tiktoken
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import OPENAI_API_KEY, LLM_MODEL
from retrieve import retrieve, TOP_K

client = OpenAI(api_key=OPENAI_API_KEY)
enc    = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        parts.append(f"[{i+1}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(parts)


def ask(question: str, k: int = TOP_K, source_filter: str = None) -> dict:
    expanded = f"{question} machine learning neural network deep learning"
    ret    = retrieve(expanded, k=k, source_filter=source_filter)
    chunks = ret["chunks"]
    ret_ms = ret["latency_ms"]

    context  = build_context(chunks)
    system = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the provided context chunks. Cite each chunk you use as [1], [2], etc. "
    "Answer based on whatever relevant information exists in the context, "
    "even if it is partial. Only say 'I could not find relevant information "
    "in the provided context' if the context is completely unrelated to the question. "
    "Do not make up any information not present in the context."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    t0   = time.time()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0
    )
    gen_ms = round((time.time() - t0) * 1000, 2)

    answer        = resp.choices[0].message.content
    prompt_tokens = resp.usage.prompt_tokens
    comp_tokens   = resp.usage.completion_tokens

    print(f"[log] retrieval: {ret_ms}ms | generation: {gen_ms}ms | "
          f"chunks: {len(chunks)} | tokens in: {prompt_tokens} out: {comp_tokens}")

    return {
        "question"         : question,
        "answer"           : answer,
        "chunks_used"      : chunks,
        "retrieval_ms"     : ret_ms,
        "generation_ms"    : gen_ms,
        "prompt_tokens"    : prompt_tokens,
        "completion_tokens": comp_tokens
    }


if __name__ == "__main__":
    q   = "What is the vanishing gradient problem?"
    out = ask(q)
    print(f"\nQ: {out['question']}")
    print(f"A: {out['answer']}")
    print(f"\nChunks used:")
    for c in out["chunks_used"]:
        print(f"  [{c['source']}] chunk {c['chunk_index']} (dist: {c['distance']})")