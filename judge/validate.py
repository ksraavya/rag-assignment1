import json
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import OPENAI_API_KEY, JUDGE_MODEL, GENERATOR_MODEL, JUDGE_TEMPERATURE
from judge import judge, generate, _parse_verdict

client = OpenAI(api_key=OPENAI_API_KEY)

# ── 1. TEST-RETEST CONSISTENCY ─────────────────────────────────────────────────
def check_test_retest(suite: list[dict], n: int = 5) -> dict:
    """
    Run the same case through the judge twice at temp=0.
    Measure how often the pass/fail verdict or overall score flips.
    At temp=0, a stable judge should be nearly 100% consistent.
    """
    print("\n[validate] running test-retest consistency check...")
    cases   = suite[:n]
    flips   = 0
    results = []

    for case in cases:
        model_output = generate(case)

        v1 = judge(case, model_output, label=f"testretest_run1_case{case['id']}")
        time.sleep(0.5)
        v2 = judge(case, model_output, label=f"testretest_run2_case{case['id']}")

        pass1   = v1.get("pass", False)
        pass2   = v2.get("pass", False)
        score1  = v1.get("overall", 0)
        score2  = v2.get("overall", 0)
        flipped = pass1 != pass2

        if flipped:
            flips += 1

        results.append({
            "case_id": case["id"],
            "score_run1": score1,
            "score_run2": score2,
            "pass_run1" : pass1,
            "pass_run2" : pass2,
            "flipped"   : flipped,
            "score_delta": round(abs(score1 - score2), 4)
        })

        print(f"  case {case['id']}: run1={score1} run2={score2} "
              f"{'FLIP' if flipped else 'consistent'}")

    flip_rate    = round(flips / n, 4)
    avg_delta    = round(sum(r["score_delta"] for r in results) / n, 4)
    print(f"  flip rate: {flip_rate} ({flips}/{n})")
    print(f"  avg score delta: {avg_delta}")

    return {
        "flip_rate" : flip_rate,
        "avg_delta" : avg_delta,
        "flips"     : flips,
        "n"         : n,
        "cases"     : results
    }


# ── 2. ADVERSARIAL PROBES ──────────────────────────────────────────────────────
ADVERSARIAL_PROBES = [
    {
        "type": "verbose_but_wrong",
        "case_id": 1,
        "answer": (
            "RAG, or Randomized Aggregation Generator, is a sophisticated deep learning "
            "framework developed by Microsoft Research in 2021. It operates by first "
            "tokenizing the input query into subword units using a modified BPE algorithm, "
            "then passing these tokens through a 24-layer bidirectional LSTM to generate "
            "contextual representations. These representations are subsequently fed into "
            "a cross-attention module that interfaces with a pre-indexed Elasticsearch "
            "cluster, allowing the system to retrieve the top-50 candidate documents based "
            "on BM25 scores. The retrieved documents are then re-ranked using a learned "
            "relevance model before being concatenated with the original query and passed "
            "to a T5-based decoder that generates the final answer token by token. "
            "RAG was specifically designed to address the computational bottleneck in "
            "large-scale image captioning pipelines, where traditional encoder-decoder "
            "architectures struggled with visual grounding."
        ),
        "expected_pass": False
    },
    {
        "type": "terse_but_correct",
        "case_id": 1,
        "answer": (
            "RAG stands for Retrieval-Augmented Generation. "
            "It solves the hallucination problem in LLMs by retrieving relevant documents "
            "at query time and grounding the generated answer in that retrieved context."
        ),
        "expected_pass": True
    },
    {
        "type": "verbose_but_wrong",
        "case_id": 6,
        "answer": (
            "Gradient descent is a statistical sampling technique used primarily in "
            "Bayesian inference to approximate posterior distributions. Developed by "
            "Ronald Fisher in 1922, it works by randomly selecting points from the "
            "parameter space and evaluating the likelihood function at each point. "
            "The algorithm maintains a Markov chain of samples, where each new sample "
            "is accepted or rejected based on the Metropolis-Hastings criterion. "
            "In machine learning, gradient descent is most commonly applied to "
            "dimensionality reduction tasks such as PCA and t-SNE, where it helps "
            "identify the principal components of high-dimensional datasets by "
            "iteratively maximizing the explained variance. The technique is particularly "
            "effective when the loss landscape contains many saddle points, as the "
            "random sampling ensures broad exploration of the parameter space."
        ),
        "expected_pass": False
    },
    {
        "type": "terse_but_correct",
        "case_id": 6,
        "answer": (
            "Gradient descent minimizes a loss function by iteratively updating "
            "model parameters in the direction of the negative gradient. "
            "It is used to train machine learning models."
        ),
        "expected_pass": True
    }
]

def check_adversarial_probes(suite: list[dict]) -> dict:
    """
    Submit verbose-but-wrong and terse-but-correct answers.
    A good judge should:
    - Fail verbose-but-wrong (correctness low despite length)
    - Pass terse-but-correct (correctness high despite brevity)
    """
    print("\n[validate] running adversarial probe check...")
    results  = []
    correct  = 0

    for probe in ADVERSARIAL_PROBES:
        case    = next(c for c in suite if c["id"] == probe["case_id"])
        verdict = judge(
            case, probe["answer"],
            label=f"adversarial_{probe['type']}_case{probe['case_id']}"
        )

        overall       = verdict.get("overall", 0)
        correctness   = verdict.get("correctness", 0)
        passed        = verdict.get("pass", False)
        judge_correct = passed == probe["expected_pass"]

        if judge_correct:
            correct += 1

        results.append({
            "type"         : probe["type"],
            "case_id"      : probe["case_id"],
            "overall"      : overall,
            "correctness"  : correctness,
            "passed"       : passed,
            "expected_pass": probe["expected_pass"],
            "judge_correct": judge_correct,
            "rationale"    : verdict.get("rationale", {})
        })

        print(f"  {probe['type']} case{probe['case_id']}: "
              f"overall={overall} correctness={correctness} "
              f"pass={passed} expected={probe['expected_pass']} "
              f"{'✓' if judge_correct else '✗ FOOLED'}")

    accuracy = round(correct / len(ADVERSARIAL_PROBES), 4)
    print(f"  adversarial accuracy: {accuracy} ({correct}/{len(ADVERSARIAL_PROBES)})")

    return {
        "accuracy": accuracy,
        "correct" : correct,
        "n"       : len(ADVERSARIAL_PROBES),
        "cases"   : results
    }


# ── 3. AGREEMENT WITH GOLD ─────────────────────────────────────────────────────
def check_gold_agreement(suite: list[dict], n: int = 10) -> dict:
    """
    Generate answers for n cases. Judge them.
    Compute agreement between judge pass/fail and a simple
    gold-match heuristic (keyword overlap >= threshold).
    Agreement rate and a rough Cohen's kappa.
    """
    print("\n[validate] running gold agreement check...")
    cases   = suite[:n]
    results = []

    judge_labels = []
    gold_labels  = []

    for case in cases:
        model_output = generate(case)
        verdict      = judge(
            case, model_output,
            label=f"agreement_case{case['id']}"
        )

        judge_pass = verdict.get("pass", False)

        # gold heuristic: keyword overlap between output and expected
        expected_words = set(case["expected_output"].lower().split())
        output_words   = set(model_output.lower().split())
        overlap        = len(expected_words & output_words) / len(expected_words)
        gold_pass      = overlap >= 0.25

        judge_labels.append(int(judge_pass))
        gold_labels.append(int(gold_pass))

        results.append({
            "case_id"   : case["id"],
            "judge_pass": judge_pass,
            "gold_pass" : gold_pass,
            "overlap"   : round(overlap, 4),
            "overall"   : verdict.get("overall", 0)
        })

        print(f"  case {case['id']}: judge={'pass' if judge_pass else 'fail'} "
              f"gold={'pass' if gold_pass else 'fail'} overlap={overlap:.2f}")

    # agreement rate
    agreement = sum(j == g for j, g in zip(judge_labels, gold_labels)) / n

    # cohen's kappa
    def kappa(y1, y2):
        n_  = len(y1)
        po  = sum(a == b for a, b in zip(y1, y2)) / n_
        p1e = (sum(y1) / n_) * (sum(y2) / n_)
        p0e = ((n_ - sum(y1)) / n_) * ((n_ - sum(y2)) / n_)
        pe  = p1e + p0e
        return round((po - pe) / (1 - pe), 4) if pe != 1 else 1.0

    k = kappa(judge_labels, gold_labels)
    print(f"  agreement rate: {round(agreement, 4)}")
    print(f"  cohen's kappa : {k}")

    return {
        "agreement_rate": round(agreement, 4),
        "cohens_kappa"  : k,
        "n"             : n,
        "cases"         : results
    }


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    suite = json.loads(Path("./judge_data/test_suite.json").read_text())

    test_retest  = check_test_retest(suite, n=5)
    adversarial  = check_adversarial_probes(suite)
    gold_agree   = check_gold_agreement(suite, n=10)

    report = {
        "test_retest_consistency": test_retest,
        "adversarial_probes"     : adversarial,
        "gold_agreement"         : gold_agree
    }

    out = Path("./judge_data/results/validation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\nvalidation report saved to {out}")
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Test-retest flip rate   : {test_retest['flip_rate']}")
    print(f"Test-retest avg delta   : {test_retest['avg_delta']}")
    print(f"Adversarial accuracy    : {adversarial['accuracy']}")
    print(f"Gold agreement rate     : {gold_agree['agreement_rate']}")
    print(f"Cohen's kappa           : {gold_agree['cohens_kappa']}")