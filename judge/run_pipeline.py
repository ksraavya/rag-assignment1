import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge.judge    import run_case
from judge.bias     import check_position_bias, check_verbosity_bias, check_sycophancy_bias, self_enhancement_note
from judge.validate import check_test_retest, check_adversarial_probes, check_gold_agreement
from judge.compare  import run_ab_comparison

RESULTS_DIR = Path("./judge_data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_full_suite(suite: list[dict]) -> dict:
    print(f"\n{'='*55}")
    print(f"RUNNING FULL JUDGE PIPELINE — {len(suite)} cases")
    print(f"{'='*55}")

    results = []
    for case in suite:
        print(f"\n[{case['id']}/{len(suite)}] {case['input'][:55]}...")
        result = run_case(case)
        v      = result["verdict"]
        print(f"  overall={v.get('overall')} pass={v.get('pass')}")
        results.append(result)
        time.sleep(0.3)

    # aggregate
    n         = len(results)
    pass_rate = round(sum(1 for r in results if r["verdict"].get("pass")) / n, 4)
    criteria  = ["correctness", "faithfulness", "completeness",
                 "instruction_following", "conciseness", "overall"]

    mean_scores = {
        c: round(sum(r["verdict"].get(c, 0) for r in results) / n, 4)
        for c in criteria
    }
    total_tokens = sum(r["verdict"].get("tokens", 0) for r in results)

    summary = {
        "n"           : n,
        "pass_rate"   : pass_rate,
        "mean_scores" : mean_scores,
        "total_tokens": total_tokens
    }

    print(f"\n{'='*55}")
    print("SUITE SUMMARY")
    print(f"{'='*55}")
    print(f"Pass rate : {pass_rate}")
    for c, s in mean_scores.items():
        print(f"  {c:<25}: {s}")
    print(f"Total tokens used: {total_tokens}")

    return {"summary": summary, "per_case": results}


if __name__ == "__main__":
    suite = json.loads(Path("./judge_data/test_suite.json").read_text())

    # 1. full suite
    print("\n>>> STEP 1: FULL SUITE JUDGING")
    suite_report = run_full_suite(suite)
    (RESULTS_DIR / "suite_report.json").write_text(
        json.dumps(suite_report, indent=2))

    # 2. bias checks
    print("\n>>> STEP 2: BIAS CHECKS")
    bias_report = {
        "position_bias"   : check_position_bias(suite, n=5),
        "verbosity_bias"  : check_verbosity_bias(suite, n=3),
        "sycophancy_bias" : check_sycophancy_bias(suite),
        "self_enhancement": self_enhancement_note()
    }
    (RESULTS_DIR / "bias_report.json").write_text(
        json.dumps(bias_report, indent=2))

    # 3. validation
    print("\n>>> STEP 3: VALIDATION")
    validation_report = {
        "test_retest_consistency": check_test_retest(suite, n=5),
        "adversarial_probes"     : check_adversarial_probes(suite),
        "gold_agreement"         : check_gold_agreement(suite, n=10)
    }
    (RESULTS_DIR / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2))

    # 4. A/B comparison
    print("\n>>> STEP 4: A/B COMPARISON")
    ab_report = run_ab_comparison(suite, n=10)
    (RESULTS_DIR / "ab_report.json").write_text(
        json.dumps(ab_report, indent=2))

    print(f"\n{'='*55}")
    print("ALL DONE — results in judge_data/results/")
    print(f"{'='*55}")
    print(f"suite_report.json     — full per-case verdicts + summary")
    print(f"bias_report.json      — position, verbosity, sycophancy")
    print(f"validation_report.json — test-retest, adversarial, agreement")
    print(f"ab_report.json        — A/B comparison + declared winner")