import json
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import OPENAI_API_KEY, JUDGE_MODEL, GENERATOR_MODEL, JUDGE_TEMPERATURE
from judge.judge import generate, _parse_verdict, _log

client = OpenAI(api_key=OPENAI_API_KEY)


def judge_with_prompt(case: dict, model_output: str, prompt_text: str, label: str) -> dict:
    """Judge a model output using a custom prompt template."""
    full_prompt = f"""{prompt_text}

Question: {case['input']}

Expected Output (reference):
{case['expected_output']}

Model Output to evaluate:
{model_output}
"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=JUDGE_TEMPERATURE
    )
    raw    = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens

    verdict = _parse_verdict(raw)
    if verdict is None:
        verdict = {
            "correctness": 0.0, "faithfulness": 0.0,
            "completeness": 0.0, "instruction_following": 0.0,
            "conciseness": 0.0, "overall": 0.0,
            "rationale": {}, "pass": False, "parse_error": True
        }

    _log({
        "case_id": case["id"], "label": label,
        "prompt": full_prompt, "raw_response": raw,
        "verdict": verdict, "tokens": tokens
    })

    verdict["tokens"] = tokens
    return verdict


def run_ab_comparison(suite: list[dict], n: int = 10) -> dict:
    """
    Compare prompt v1 (basic) vs v2 (rubric + anchors) on n cases.
    Same generated answer, different judge prompts.
    Declares a winner based on score spread and pass rate.
    """
    print("\n[compare] running A/B comparison (v1 vs v2 prompt)...")

    v1_prompt = Path("./judge_data/prompts/v1.txt").read_text()
    v2_prompt = Path("./judge_data/prompts/v2.txt").read_text()

    cases   = suite[:n]
    results = []

    v1_scores  = []
    v2_scores  = []
    v1_passes  = 0
    v2_passes  = 0
    v2_wins    = 0
    v1_wins    = 0
    ties       = 0

    for case in cases:
        print(f"  case {case['id']}...", end=" ", flush=True)
        model_output = generate(case)

        v1 = judge_with_prompt(
            case, model_output, v1_prompt,
            label=f"ab_v1_case{case['id']}"
        )
        v2 = judge_with_prompt(
            case, model_output, v2_prompt,
            label=f"ab_v2_case{case['id']}"
        )

        s1 = v1.get("overall", 0)
        s2 = v2.get("overall", 0)
        p1 = v1.get("pass", False)
        p2 = v2.get("pass", False)

        v1_scores.append(s1)
        v2_scores.append(s2)
        if p1: v1_passes += 1
        if p2: v2_passes += 1

        # score spread per case (higher spread = better discrimination)
        spread1 = _score_spread(v1)
        spread2 = _score_spread(v2)

        if spread2 > spread1:
            v2_wins += 1
            case_winner = "v2"
        elif spread1 > spread2:
            v1_wins += 1
            case_winner = "v1"
        else:
            ties += 1
            case_winner = "tie"

        print(f"v1={s1} v2={s2} spread: v1={spread1:.3f} v2={spread2:.3f} → {case_winner}")

        results.append({
            "case_id"    : case["id"],
            "v1_overall" : s1,
            "v2_overall" : s2,
            "v1_pass"    : p1,
            "v2_pass"    : p2,
            "v1_spread"  : spread1,
            "v2_spread"  : spread2,
            "case_winner": case_winner
        })

        time.sleep(0.3)

    avg_v1 = round(sum(v1_scores) / n, 4)
    avg_v2 = round(sum(v2_scores) / n, 4)
    avg_spread_v1 = round(sum(r["v1_spread"] for r in results) / n, 4)
    avg_spread_v2 = round(sum(r["v2_spread"] for r in results) / n, 4)

    # declare winner: better discrimination (spread) is the key metric
    # pass rate as tiebreaker
    if avg_spread_v2 > avg_spread_v1:
        winner = "v2"
        reason = "higher average score spread — better discrimination between criteria"
    elif avg_spread_v1 > avg_spread_v2:
        winner = "v1"
        reason = "higher average score spread"
    else:
        winner = "v2" if v2_passes >= v1_passes else "v1"
        reason = "higher pass rate (spread was equal)"

    summary = {
        "n"             : n,
        "v1_mean_score" : avg_v1,
        "v2_mean_score" : avg_v2,
        "v1_pass_rate"  : round(v1_passes / n, 4),
        "v2_pass_rate"  : round(v2_passes / n, 4),
        "v1_avg_spread" : avg_spread_v1,
        "v2_avg_spread" : avg_spread_v2,
        "v2_wins"       : v2_wins,
        "v1_wins"       : v1_wins,
        "ties"          : ties,
        "winner"        : winner,
        "winner_reason" : reason,
        "cases"         : results
    }

    print(f"\n  v1 mean={avg_v1} pass_rate={summary['v1_pass_rate']} spread={avg_spread_v1}")
    print(f"  v2 mean={avg_v2} pass_rate={summary['v2_pass_rate']} spread={avg_spread_v2}")
    print(f"\n  WINNER: {winner.upper()} — {reason}")

    return summary


def _score_spread(verdict: dict) -> float:
    """Std deviation of per-criterion scores — higher = better discrimination."""
    scores = [
        verdict.get("correctness", 0),
        verdict.get("faithfulness", 0),
        verdict.get("completeness", 0),
        verdict.get("instruction_following", 0),
        verdict.get("conciseness", 0)
    ]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return variance ** 0.5


if __name__ == "__main__":
    suite  = json.loads(Path("./judge_data/test_suite.json").read_text())
    result = run_ab_comparison(suite, n=10)

    out = Path("./judge_data/results/ab_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nA/B report saved to {out}")