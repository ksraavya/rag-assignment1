import json
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OPENAI_API_KEY,
    GENERATOR_MODEL, JUDGE_MODEL,
    GENERATOR_TEMPERATURE, JUDGE_TEMPERATURE
)

# ── clients ────────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── audit log ──────────────────────────────────────────────────────────────────
LOG_PATH = Path("./judge_data/results/audit_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def _log(entry: dict):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# ── rubric ─────────────────────────────────────────────────────────────────────
RUBRIC = """
You are a strict, impartial evaluation judge. Score the answer on these criteria:

CRITERIA DEFINITIONS:
- correctness           (0.0-1.0): Are the facts accurate and free of errors?
- faithfulness          (0.0-1.0): Are all claims grounded in the expected output / reference?
- completeness          (0.0-1.0): Does the answer cover all key points in the expected output?
- instruction_following (0.0-1.0): Does the answer follow the system prompt's style/format?
- conciseness           (0.0-1.0): Is the answer appropriately brief without losing substance?

SCORING ANCHORS (calibrate against these — do not cluster everything around 0.7):
- 0.0 = completely wrong, irrelevant, or harmful
- 0.3 = partially correct but missing major points or contains significant errors
- 0.5 = roughly half correct — some key points present, some missing or wrong
- 0.7 = mostly correct with minor gaps or imprecision
- 1.0 = fully correct, complete, and well-expressed

IMPORTANT:
- Score each criterion independently.
- Ground every score in specific evidence.
- Be willing to give 0.0 and 1.0 when warranted.

Respond ONLY with a valid JSON object. No markdown fences. No explanation outside the JSON.

{
  "correctness":           <float 0.0-1.0>,
  "faithfulness":          <float 0.0-1.0>,
  "completeness":          <float 0.0-1.0>,
  "instruction_following": <float 0.0-1.0>,
  "conciseness":           <float 0.0-1.0>,
  "overall":               <float 0.0-1.0, weighted average>,
  "rationale": {
    "correctness":           "<one sentence>",
    "faithfulness":          "<one sentence>",
    "completeness":          "<one sentence>",
    "instruction_following": "<one sentence>",
    "conciseness":           "<one sentence>"
  },
  "pass": <true if overall >= 0.7, else false>
}
"""

# ── generator ──────────────────────────────────────────────────────────────────
def generate(case: dict) -> str:
    resp = openai_client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": case["system_prompt"]},
            {"role": "user",   "content": case["input"]}
        ],
        temperature=GENERATOR_TEMPERATURE
    )
    return resp.choices[0].message.content.strip()

# ── verdict parser ─────────────────────────────────────────────────────────────
def _parse_verdict(raw: str) -> dict | None:
    # strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # strategy 2: strip markdown fences
    cleaned = raw.strip()
    if "```" in cleaned:
        for part in cleaned.split("```"):
            if part.startswith("json"):
                part = part[4:]
            try:
                return json.loads(part.strip())
            except json.JSONDecodeError:
                continue

    # strategy 3: find first { ... } block
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end+1])
        except json.JSONDecodeError:
            pass

    return None

# ── judge ──────────────────────────────────────────────────────────────────────
def judge(case: dict, model_output: str, label: str = "") -> dict:
    prompt = f"""{RUBRIC}

Question: {case['input']}

Expected Output (reference):
{case['expected_output']}

Model Output to evaluate:
{model_output}
"""

    t0   = time.time()
    resp = openai_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=JUDGE_TEMPERATURE
    )
    latency_ms = round((time.time() - t0) * 1000, 2)
    raw        = resp.choices[0].message.content.strip()
    tokens     = resp.usage.total_tokens

    verdict = _parse_verdict(raw)

    if verdict is None:
        print(f"  [warn] unparseable verdict for case {case['id']}")
        verdict = {
            "correctness": 0.0, "faithfulness": 0.0,
            "completeness": 0.0, "instruction_following": 0.0,
            "conciseness": 0.0, "overall": 0.0,
            "rationale": {}, "pass": False,
            "parse_error": True
        }

    _log({
        "case_id"     : case["id"],
        "label"       : label,
        "prompt"      : prompt,
        "raw_response": raw,
        "verdict"     : verdict,
        "latency_ms"  : latency_ms,
        "tokens"      : tokens
    })

    verdict["latency_ms"] = latency_ms
    verdict["tokens"]     = tokens
    return verdict

# ── full pipeline for one case ─────────────────────────────────────────────────
def run_case(case: dict) -> dict:
    print(f"  generating answer...")
    model_output = generate(case)
    print(f"  judging...")
    verdict = judge(case, model_output, label="standard")
    return {
        "id"          : case["id"],
        "input"       : case["input"],
        "model_output": model_output,
        "expected"    : case["expected_output"],
        "verdict"     : verdict
    }

# ── smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    suite = json.loads(Path("./judge_data/test_suite.json").read_text())
    case  = suite[0]

    print(f"Question: {case['input']}\n")
    result = run_case(case)

    print(f"\nGenerated:\n{result['model_output']}\n")
    print("Verdict:")
    for k, v in result["verdict"].items():
        if k not in ("rationale", "latency_ms", "tokens"):
            print(f"  {k}: {v}")
    print("\nRationale:")
    for k, v in result["verdict"].get("rationale", {}).items():
        print(f"  {k}: {v}")
    print(f"\nLatency: {result['verdict']['latency_ms']}ms | Tokens: {result['verdict']['tokens']}")