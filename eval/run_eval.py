import json
import sys
import time
import math
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from answer import ask
from config import OPENAI_API_KEY, TOP_K

client = OpenAI(api_key=OPENAI_API_KEY)

# ── load questions ─────────────────────────────────────────────────────────────
def load_questions(path: str = "./eval/questions.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

# ── retrieval metrics ──────────────────────────────────────────────────────────
def hit_rate(chunks: list[dict], relevant_source: str) -> float:
    return float(any(c["source"] == relevant_source for c in chunks))

def mrr(chunks: list[dict], relevant_source: str) -> float:
    for i, c in enumerate(chunks):
        if c["source"] == relevant_source:
            return 1.0 / (i + 1)
    return 0.0

def ndcg(chunks: list[dict], relevant_source: str, k: int) -> float:
    gains = [1.0 if c["source"] == relevant_source else 0.0 for c in chunks[:k]]
    dcg   = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    # ideal: all relevant at top
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(sum(1 for g in gains if g), k)))
    return dcg / ideal if ideal > 0 else 0.0

def context_precision(chunks: list[dict], relevant_source: str) -> float:
    relevant = sum(1 for c in chunks if c["source"] == relevant_source)
    return relevant / len(chunks) if chunks else 0.0

# ── answer metrics (llm-as-judge) ──────────────────────────────────────────────
def judge_answer(question: str, answer: str, context: str, gold: str) -> dict:
    prompt = f"""You are an evaluation judge for a RAG system. Score the answer on two criteria.

Question: {question}

Retrieved Context:
{context}

Gold Answer:
{gold}

Generated Answer:
{answer}

Score each criterion from 0.0 to 1.0 and respond ONLY with valid JSON, no explanation outside the JSON:
{{
  "faithfulness": <0.0-1.0, is every claim in the answer supported by the retrieved context?>,
  "answer_relevance": <0.0-1.0, does the answer address the question asked?>,
  "faithfulness_reason": "<one sentence>",
  "answer_relevance_reason": "<one sentence>"
}}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip()

    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw.strip())

# ── main eval loop ─────────────────────────────────────────────────────────────
def run_eval(questions_path: str = "./eval/questions.json"):
    questions = load_questions(questions_path)
    results   = []

    print(f"running eval on {len(questions)} questions...\n")

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        source   = q["source"]
        gold     = q["gold_answer"]

        print(f"[{qid}/{len(questions)}] {question[:60]}...")

        # get answer from RAG system
        out    = ask(question)
        chunks = out["chunks_used"]
        answer = out["answer"]

        # retrieval metrics
        hr  = hit_rate(chunks, source)
        rr  = mrr(chunks, source)
        ng  = ndcg(chunks, source, k=TOP_K)
        cp  = context_precision(chunks, source)

        # build context string for judge
        context_str = "\n\n".join(
            f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
        )

        # answer metrics
        try:
            scores = judge_answer(question, answer, context_str, gold)
            faith  = scores["faithfulness"]
            relev  = scores["answer_relevance"]
            faith_r = scores.get("faithfulness_reason", "")
            relev_r = scores.get("answer_relevance_reason", "")
        except Exception as e:
            print(f"  [warn] judge failed: {e}")
            faith, relev, faith_r, relev_r = 0.0, 0.0, "", ""

        print(f"  hit={hr} mrr={rr:.2f} ndcg={ng:.2f} cp={cp:.2f} "
              f"faith={faith:.2f} relev={relev:.2f}")

        results.append({
            "id"                 : qid,
            "question"           : question,
            "gold_answer"        : gold,
            "generated_answer"   : answer,
            "relevant_source"    : source,
            "hit_rate"           : hr,
            "mrr"                : rr,
            "ndcg"               : ng,
            "context_precision"  : cp,
            "faithfulness"       : faith,
            "answer_relevance"   : relev,
            "faithfulness_reason": faith_r,
            "relevance_reason"   : relev_r,
            "retrieval_ms"       : out["retrieval_ms"],
            "generation_ms"      : out["generation_ms"],
            "prompt_tokens"      : out["prompt_tokens"],
            "completion_tokens"  : out["completion_tokens"],
        })

        time.sleep(0.5)  # avoid rate limits

    # ── aggregate ──────────────────────────────────────────────────────────────
    n = len(results)
    summary = {
        "total_questions"      : n,
        "hit_rate"             : round(sum(r["hit_rate"] for r in results) / n, 4),
        "mrr"                  : round(sum(r["mrr"] for r in results) / n, 4),
        "ndcg"                 : round(sum(r["ndcg"] for r in results) / n, 4),
        "context_precision"    : round(sum(r["context_precision"] for r in results) / n, 4),
        "faithfulness"         : round(sum(r["faithfulness"] for r in results) / n, 4),
        "answer_relevance"     : round(sum(r["answer_relevance"] for r in results) / n, 4),
        "avg_retrieval_ms"     : round(sum(r["retrieval_ms"] for r in results) / n, 2),
        "avg_generation_ms"    : round(sum(r["generation_ms"] for r in results) / n, 2),
        "total_prompt_tokens"  : sum(r["prompt_tokens"] for r in results),
        "total_completion_tokens": sum(r["completion_tokens"] for r in results),
    }

    # ── save results ───────────────────────────────────────────────────────────
    output = {"summary": summary, "per_question": results}
    out_path = Path("./eval/results/eval_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # ── print summary ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Hit Rate         : {summary['hit_rate']}")
    print(f"MRR              : {summary['mrr']}")
    print(f"nDCG@{TOP_K}          : {summary['ndcg']}")
    print(f"Context Precision: {summary['context_precision']}")
    print(f"Faithfulness     : {summary['faithfulness']}")
    print(f"Answer Relevance : {summary['answer_relevance']}")
    print(f"Avg Retrieval ms : {summary['avg_retrieval_ms']}")
    print(f"Avg Generation ms: {summary['avg_generation_ms']}")
    print(f"Total tokens used: {summary['total_prompt_tokens'] + summary['total_completion_tokens']}")
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    run_eval()