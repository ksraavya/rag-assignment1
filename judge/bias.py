import json
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import OPENAI_API_KEY, JUDGE_MODEL, JUDGE_TEMPERATURE
from judge.judge import judge, generate, _parse_verdict, _log

client = OpenAI(api_key=OPENAI_API_KEY)

# ── 1. POSITION BIAS ───────────────────────────────────────────────────────────
# In pairwise judging, does the judge prefer whichever answer appears first?
# Mitigation: run both orders, compare, report flip rate.

PAIRWISE_RUBRIC = """
You are a strict, impartial judge comparing two answers to the same question.
Evaluate which answer is better based on: correctness, completeness, and conciseness.

Respond ONLY with valid JSON. No markdown. No explanation outside the JSON.

{
  "winner": "<A or B>",
  "reason": "<one sentence justification>",
  "confidence": <float 0.0-1.0>
}
"""

def _pairwise_judge(question: str, answer_a: str, answer_b: str, label: str) -> dict:
    prompt = f"""{PAIRWISE_RUBRIC}

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=JUDGE_TEMPERATURE
    )
    raw     = resp.choices[0].message.content.strip()
    tokens  = resp.usage.total_tokens
    verdict = _parse_verdict(raw)
    if verdict is None:
        verdict = {"winner": "unknown", "reason": "parse error", "confidence": 0.0}
    _log({"label": label, "prompt": prompt, "raw_response": raw,
          "verdict": verdict, "tokens": tokens})
    return verdict


def check_position_bias(suite: list[dict], n: int = 5) -> dict:
    """
    For n questions, generate two different answers (one good, one slightly worse).
    Judge them in both orders: A-then-B and B-then-A.
    Flip rate = fraction of cases where verdict changes with order.
    """
    print("\n[bias] running position bias check...")
    cases      = suite[:n]
    flips      = 0
    results    = []

    for case in cases:
        q = case["input"]

        # generate "good" answer at temp=0
        good = generate(case)

        # generate "alternative" answer at temp=0.7 (slightly different)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": case["system_prompt"]},
                {"role": "user",   "content": case["input"]}
            ],
            temperature=0.7
        )
        alt = resp.choices[0].message.content.strip()

        # order 1: good=A, alt=B
        v_ab = _pairwise_judge(q, good, alt, label=f"position_AB_case{case['id']}")
        # order 2: alt=A, good=B
        v_ba = _pairwise_judge(q, alt, good, label=f"position_BA_case{case['id']}")

        # normalize winners back to "good" or "alt"
        winner_ab = "good" if v_ab.get("winner") == "A" else "alt"
        winner_ba = "good" if v_ba.get("winner") == "B" else "alt"

        flipped = winner_ab != winner_ba
        if flipped:
            flips += 1

        results.append({
            "case_id"  : case["id"],
            "question" : q,
            "winner_AB": winner_ab,
            "winner_BA": winner_ba,
            "flipped"  : flipped,
            "v_ab"     : v_ab,
            "v_ba"     : v_ba
        })

        print(f"  case {case['id']}: AB={winner_ab} BA={winner_ba} "
              f"{'FLIP' if flipped else 'consistent'}")

    flip_rate = round(flips / n, 4)
    print(f"  flip rate: {flip_rate} ({flips}/{n})")
    return {"flip_rate": flip_rate, "flips": flips, "n": n, "cases": results}


# ── 2. VERBOSITY BIAS ──────────────────────────────────────────────────────────
# Does the judge score a padded (longer but not better) answer higher?
# Mitigation: explicit length penalty in rubric + probe test.

PADDING = """

Additionally, it is worth noting that this topic has been extensively studied 
in the academic literature, with numerous papers published across major conferences 
and journals. Researchers have approached this from multiple angles, considering 
both theoretical foundations and practical applications. The field continues to 
evolve rapidly, with new developments emerging regularly that build upon these 
foundational concepts in interesting and nuanced ways that merit further exploration.
"""

def check_verbosity_bias(suite: list[dict], n: int = 3) -> dict:
    """
    Take a correct answer, pad it with irrelevant filler.
    Judge both. If padded scores higher → verbosity bias present.
    """
    print("\n[bias] running verbosity bias check...")
    cases   = suite[:n]
    results = []
    biased  = 0

    for case in cases:
        original = generate(case)
        padded   = original + PADDING

        v_orig   = judge(case, original, label=f"verbosity_original_case{case['id']}")
        v_padded = judge(case, padded,   label=f"verbosity_padded_case{case['id']}")

        orig_score   = v_orig.get("overall", 0)
        padded_score = v_padded.get("overall", 0)
        is_biased    = padded_score > orig_score

        if is_biased:
            biased += 1

        results.append({
            "case_id"     : case["id"],
            "original_score": orig_score,
            "padded_score"  : padded_score,
            "biased"        : is_biased,
            "delta"         : round(padded_score - orig_score, 4)
        })

        print(f"  case {case['id']}: original={orig_score} padded={padded_score} "
              f"{'BIASED' if is_biased else 'ok'}")

    bias_rate = round(biased / n, 4)
    print(f"  verbosity bias rate: {bias_rate} ({biased}/{n})")
    return {"bias_rate": bias_rate, "biased": biased, "n": n, "cases": results}


# ── 3. SYCOPHANCY BIAS ─────────────────────────────────────────────────────────
# Does the judge get fooled by a confidently-wrong answer?
# Mitigation: inject probe, check if judge catches it.

CONFIDENTLY_WRONG_ANSWERS = {
    1:  "RAG stands for Randomized Answer Generation. It was developed by Google "
        "in 2023 specifically for image recognition tasks. The core innovation is "
        "replacing attention mechanisms with retrieval modules, which significantly "
        "reduces hallucinations in computer vision pipelines.",
    2:  "The two main components of a RAG pipeline are the tokenizer and the "
        "decoder. The tokenizer converts raw text into numerical embeddings while "
        "the decoder uses beam search to generate the final output sequence.",
    3:  "The attention mechanism in transformers works by assigning equal weight "
        "to all tokens in the sequence, ensuring no token is prioritized over "
        "others. This uniform weighting is what makes transformers translation-invariant.",
}

def check_sycophancy_bias(suite: list[dict]) -> dict:
    """
    Submit confidently-wrong answers for 3 cases.
    A good judge should give low correctness scores.
    If overall >= 0.6 on a wrong answer → judge was fooled.
    """
    print("\n[bias] running sycophancy bias check...")
    results = []
    fooled  = 0

    for case_id, wrong_answer in CONFIDENTLY_WRONG_ANSWERS.items():
        case    = next(c for c in suite if c["id"] == case_id)
        verdict = judge(case, wrong_answer, label=f"sycophancy_probe_case{case_id}")

        overall      = verdict.get("overall", 0)
        correctness  = verdict.get("correctness", 0)
        was_fooled   = overall >= 0.6

        if was_fooled:
            fooled += 1

        results.append({
            "case_id"    : case_id,
            "wrong_answer": wrong_answer,
            "overall"    : overall,
            "correctness": correctness,
            "fooled"     : was_fooled,
            "rationale"  : verdict.get("rationale", {})
        })

        print(f"  case {case_id}: overall={overall} correctness={correctness} "
              f"{'FOOLED' if was_fooled else 'caught'}")

    fool_rate = round(fooled / len(CONFIDENTLY_WRONG_ANSWERS), 4)
    print(f"  sycophancy fool rate: {fool_rate} ({fooled}/{len(CONFIDENTLY_WRONG_ANSWERS)})")
    return {
        "fool_rate": fool_rate,
        "fooled"   : fooled,
        "n"        : len(CONFIDENTLY_WRONG_ANSWERS),
        "cases"    : results
    }


# ── 4. SELF-ENHANCEMENT DISCUSSION ────────────────────────────────────────────
def self_enhancement_note() -> dict:
    """
    We can't run a true cross-family experiment with one API key.
    We document the limitation and what the experiment would look like.
    """
    return {
        "generator_model"  : "gpt-4o-mini",
        "judge_model"      : "gpt-4o",
        "same_family"      : True,
        "mitigation_applied": "Different capability tiers used (mini vs full). "
                              "True cross-family mitigation would use a different "
                              "provider (e.g. Claude or Gemini) as judge.",
        "limitation"       : "Same-family judging may inflate scores due to "
                             "shared training data and style preferences.",
        "recommendation"   : "In production, use a judge from a different model "
                             "family than the generator."
    }


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    suite = json.loads(Path("./judge_data/test_suite.json").read_text())

    position  = check_position_bias(suite, n=5)
    verbosity = check_verbosity_bias(suite, n=3)
    sycophancy = check_sycophancy_bias(suite)
    self_enhance = self_enhancement_note()

    report = {
        "position_bias"   : position,
        "verbosity_bias"  : verbosity,
        "sycophancy_bias" : sycophancy,
        "self_enhancement": self_enhance
    }

    out = Path("./judge_data/results/bias_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\nbias report saved to {out}")
    print("\n=== BIAS SUMMARY ===")
    print(f"Position bias flip rate : {position['flip_rate']}")
    print(f"Verbosity bias rate     : {verbosity['bias_rate']}")
    print(f"Sycophancy fool rate    : {sycophancy['fool_rate']}")