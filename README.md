# Cost Efficient RAG and LLM as Judge Evaluation

Two-problem submission covering a cost-efficient RAG application and an LLM-as-judge evaluation pipeline with bias detection.

---

## Table of Contents

- [Problem 1 — Cost-Efficient RAG Application](#problem-1--cost-efficient-rag-application)
  - [Architecture](#architecture)
  - [Setup & Run Instructions](#setup--run-instructions)
  - [Vector Store: ChromaDB](#vector-store-chromadb-embedded)
  - [Evaluation Results](#evaluation-results)
  - [Design Decisions & Trade-offs](#design-decisions--trade-offs)
- [Problem 2 — LLM-as-Judge Evaluation Pipeline](#problem-2--llm-as-judge-evaluation-pipeline)
  - [Architecture](#architecture-1)
  - [Setup & Run Instructions](#setup--run-instructions-1)
  - [Judging Mode & Rubric](#judging-mode--rubric)
  - [Evaluation Results](#evaluation-results-1)
  - [Bias Handling](#bias-handling)
  - [Judge Validation](#judge-validation)
  - [A/B Comparison](#ab-comparison)
  - [Design Decisions & Trade-offs](#design-decisions--trade-offs-1)
- [Repository Structure](#repository-structure)

---

# Problem 1 — Cost-Efficient RAG Application

A retrieval-augmented generation (RAG) QA service over a Wikipedia ML corpus, backed by ChromaDB (embedded), evaluated across retrieval quality, answer quality, latency, and cost.

## Introduction

This project implements a cost-efficient retrieval-augmented generation (RAG) system as part of an Applied AI/ML Engineering assignment. The system answers natural language questions over a document corpus by retrieving semantically relevant chunks from a local vector store and generating grounded, cited answers using an LLM. The core argument: managed vector databases like Pinecone charge for always-on pods that scale with stored vectors — a significant cost at moderate scale. This project uses ChromaDB (embedded, self-hosted) as a credible alternative, and backs that claim with retrieval metrics, answer quality scores, latency measurements, and a cost comparison across three orders of magnitude.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│                                                                 │
│  docs/ (HTML/PDF/MD)                                            │
│       │                                                         │
│       ▼                                                         │
│  load_file()  ──►  BeautifulSoup / pypdf / plain text           │
│       │                                                         │
│       ▼                                                         │
│  RecursiveCharacterTextSplitter                                 │
│  chunk_size=512, chunk_overlap=50                               │
│       │                                                         │
│       ▼                                                         │
│  MD5(chunk_text) → chunk_id  (idempotency key)                  │
│       │                                                         │
│       ▼                                                         │
│  OpenAI text-embedding-3-small (1536-dim)                       │
│       │                                                         │
│       ▼                                                         │
│  ChromaDB PersistentClient  ──►  collection.upsert()            │
│  metadata: {source, filename, chunk_index}                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                          │
│                                                                 │
│  User question                                                  │
│       │                                                         │
│       ▼                                                         │
│  Query expansion  (+  "machine learning neural network")        │
│       │                                                         │
│       ▼                                                         │
│  OpenAI text-embedding-3-small → query vector                   │
│       │                                                         │
│       ▼                                                         │
│  ChromaDB  collection.query()  top-k  (default k=5)            │
│  optional metadata filter: {source: <filename_stem>}           │
│       │                                                         │
│       ├──► chunks found?                                        │
│       │         │                                               │
│       │    YES  ▼                                               │
│       │    Build context  [1]...[k]  + original question        │
│       │         │                                               │
│       │         ▼                                               │
│       │    GPT-4o-mini  (temp=0)  → cited answer               │
│       │                                                         │
│       └──► NO relevant context → "I could not find..."         │
│                                  (no hallucination)             │
│                                                                 │
│  Log: retrieval_ms, generation_ms, chunk_count, token_usage     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup & Run Instructions

### Prerequisites
- Python 3.11+
- Windows / macOS / Linux
- OpenAI API key (embeddings + LLM)
- No external services required — ChromaDB runs embedded

### Environment variables
```
OPENAI_API_KEY=        # required
CHROMA_PATH=           # default: ./chroma_store
CHUNK_SIZE=            # default: 512
CHUNK_OVERLAP=         # default: 50
TOP_K=                 # default: 5
EMBED_MODEL=           # default: text-embedding-3-small
LLM_MODEL=             # default: gpt-4o-mini
```
Copy `.env.example` to `.env` and fill in your key. Never commit `.env`.

### Install
```bash
git clone <your-repo-url>
cd rag-assign-1
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Ingest corpus
```bash
# default: reads all files from ./docs
python src/ingest.py

# or specify a custom directory
python src/ingest.py ./my_docs
```
Chunk size and overlap are read from `.env`. Re-running is safe — idempotent by design.

### Run a query (HTTP endpoint)
```bash
uvicorn src.api:app --reload
```

```bash
# health check
GET http://127.0.0.1:8000/health

# query
POST http://127.0.0.1:8000/query
Content-Type: application/json

{
  "question": "What is the vanishing gradient problem?",
  "k": 5,
  "source_filter": null
}
```

Response includes: `answer`, `chunks_used` (with citations), `retrieval_ms`, `generation_ms`, `prompt_tokens`, `completion_tokens`.

Interactive docs at `http://127.0.0.1:8000/docs`.

### Run evaluation
```bash
python eval/run_eval.py
# results saved to eval/results/eval_results.json
```

---

## Vector Store: ChromaDB (embedded)

ChromaDB runs as a library inside the Python process — no server, no Docker, no infra. Data is persisted to disk as files. Chosen because:
- Zero infrastructure cost (just disk space)
- Python-native, zero-config setup
- Supports metadata filtering out of the box
- Sufficient for single-node, moderate-scale workloads

Trade-offs accepted: no horizontal scaling, no multi-node replication, not suitable for concurrent high-throughput production traffic.

---

## Evaluation Results

Evaluated on 20 questions across 10 Wikipedia ML articles. All metrics computed by `eval/run_eval.py`.

### Retrieval Metrics (k=5)

| Metric | Value | How computed |
|--------|-------|-------------|
| Hit Rate | 1.0 | Fraction of questions where ≥1 retrieved chunk matched the ground-truth source |
| MRR | 0.9167 | Mean reciprocal rank of first relevant chunk across all questions |
| nDCG@5 | 0.9244 | Normalized discounted cumulative gain at k=5 |
| Context Precision | 0.81 | Fraction of retrieved chunks that matched the ground-truth source |

A retrieved chunk is defined as "relevant" if its `source` metadata field matches the ground-truth source for that question (e.g. `recurrent-neural-network` for questions about RNNs).

### Answer Quality

| Metric | Score | Method | Notes |
|--------|-------|--------|-------|
| Faithfulness | 0.95 | LLM-as-judge (gpt-4o-mini) | Is every claim in the answer supported by retrieved chunks? |
| Answer Relevance | 0.95 | LLM-as-judge (gpt-4o-mini) | Does the answer address the question asked? |
| EM | N/A | — | Gold answers are descriptive, not extractive |
| F1 | N/A | — | Gold answers are descriptive, not extractive |

**Note on judge bias:** The same model family (gpt-4o-mini) is used as both generator and judge in Problem 1, which introduces self-enhancement bias. Faithfulness and relevance scores should be interpreted with this in mind. Problem 2 addresses this directly.

### Cost Comparison

Assumptions: `text-embedding-3-small` produces 1536-dim float32 vectors (~6KB/vector). Pinecone serverless pricing as of mid-2025 ($0.033/GB storage + $8/1M read units). ChromaDB cost = disk storage only on a $10/mo VPS.

| Vectors | ChromaDB ($/mo) | Pinecone Serverless ($/mo) | Notes |
|---------|----------------|---------------------------|-------|
| 100K | ~$0.10 | ~$2–5 | ChromaDB: ~0.6GB disk on shared VPS |
| 1M | ~$1.00 | ~$20–50 | ChromaDB: ~6GB disk |
| 10M | ~$10.00 | ~$200–500 | ChromaDB: ~60GB disk; single-node limit approaches |

ChromaDB embedded is essentially free at small-to-medium scale. The cost difference widens dramatically at scale — but so do the operational trade-offs.

### Latency

| Metric | Value | Conditions |
|--------|-------|-----------|
| Avg Retrieval | 745ms | k=5, 1085 vectors, OpenAI embedding API call included |
| Avg Generation | 2358ms | gpt-4o-mini, temp=0, ~700 prompt tokens |
| Total tokens (20 Qs) | 15110 | ~755 tokens/query average |

Most retrieval latency comes from the OpenAI embedding API call (~600–700ms), not ChromaDB lookup itself (sub-millisecond locally). p95 retrieval is approximately 1.2–1.5× the average given API variance.

---

## Design Decisions & Trade-offs

### Chunking strategy
Default: `chunk_size=512`, `chunk_overlap=50`, using `RecursiveCharacterTextSplitter`. The splitter tries to break on paragraphs, then sentences, then words — preserving semantic boundaries where possible. Overlap of 50 tokens ensures answers that span chunk boundaries are still captured. Larger chunks (1024+) reduced retrieval precision in early testing — too much irrelevant content per chunk diluted the embedding signal. Smaller chunks (256) caused answers to be incomplete since a single chunk didn't contain enough context.

### Embedding model
`text-embedding-3-small` (1536-dim, $0.02/1M tokens). Chosen over `text-embedding-3-large` (3072-dim, $0.13/1M tokens) because the quality difference is marginal for a focused domain corpus while the cost and storage difference is significant at scale. Dimensionality is recorded in metadata for reproducibility.

### Idempotent re-ingest
Each chunk is identified by `MD5(chunk_text)` used as the ChromaDB document ID. `collection.upsert()` with the same ID overwrites the existing vector — so re-running ingestion on an unchanged corpus produces zero new vectors. Verified: running ingest twice yields identical collection count (1085).

### No-hallucination guard
System prompt explicitly instructs the LLM to answer only from provided context chunks and to respond with "I could not find relevant information in the provided context" if the context is insufficient. Temperature is set to 0 for deterministic outputs.

### Query expansion
Short or ambiguous queries (e.g. "What is the vanishing gradient problem?") can semantically match the wrong source due to vocabulary overlap (gradient → gradient-descent article). A lightweight query expansion appends "machine learning neural network deep learning" to the embedded query — not to the question shown to the LLM. This improved hit rate significantly without affecting answer quality.

### Trade-offs accepted and when to switch back
ChromaDB embedded is the right choice when: corpus fits on one machine, query volume is moderate, and operational simplicity matters. You would switch to a managed DB (Pinecone, Weaviate Cloud) when: you need multi-region replication, horizontal read scaling, SLA guarantees, or your vector count exceeds ~50M. The break-even cost point where managed overhead becomes worthwhile is roughly 5–10M vectors with sustained high query volume.

### Was retrieval or generation the weak link?
Retrieval — specifically embedding quality on short ambiguous queries — was the primary weak link. Answer quality (faithfulness, relevance) was high once the right chunks were retrieved. This suggests the system would benefit more from better retrieval (hybrid search, re-ranking) than from a stronger LLM.

---

# Problem 2 — LLM-as-Judge Evaluation Pipeline

A judging pipeline that scores LLM outputs using a structured rubric, with concrete bias detection, mitigation, and measurement.

## Introduction

Teams use strong LLMs to automatically score model outputs when human review cannot scale. But judges are themselves biased — they prefer longer answers, favour outputs from their own model family, and cluster scores in the middle. This pipeline builds a reference-based pointwise judge, names each bias explicitly, mitigates it in code, and measures whether the mitigation worked.

**Generator:** `gpt-4o-mini` (OpenAI) — produces answers to test questions.
**Judge:** `gpt-4o` (OpenAI) — scores those answers against gold references.

Using different capability tiers within the same family partially mitigates self-enhancement bias. True cross-family judging (e.g. Claude judging GPT outputs) would be the ideal setup and is discussed in the design decisions section.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     JUDGING PIPELINE                            │
│                                                                 │
│  test_suite.json  {input, system_prompt, expected_output}       │
│       │                                                         │
│       ▼                                                         │
│  Generator (gpt-4o-mini)  →  model_output                      │
│       │                                                         │
│       ▼                                                         │
│  Judge prompt construction:                                     │
│    rubric + scoring anchors + question + expected + output      │
│       │                                                         │
│       ▼                                                         │
│  Judge (gpt-4o)  →  raw JSON verdict                           │
│       │                                                         │
│       ▼                                                         │
│  _parse_verdict()  ──► strategy 1: direct JSON parse            │
│                    ──► strategy 2: strip markdown fences        │
│                    ──► strategy 3: extract { } block            │
│                    ──► fallback: zero scores + parse_error flag │
│       │                                                         │
│       ▼                                                         │
│  audit_log.jsonl  ←── every prompt + raw response logged        │
│       │                                                         │
│       ▼                                                         │
│  Per-case verdict: {correctness, faithfulness, completeness,    │
│    instruction_following, conciseness, overall, pass, rationale}│
│       │                                                         │
│       ▼                                                         │
│  Aggregate → suite_report.json                                  │
│                                                                 │
│  Bias checks  →  bias_report.json                               │
│  ├── position bias: run A/B in both orders, measure flip rate   │
│  ├── verbosity: probe padded answer, check score delta          │
│  └── sycophancy: inject confidently-wrong answer                │
│                                                                 │
│  Validation  →  validation_report.json                          │
│  ├── test-retest: same case twice, measure flip rate            │
│  ├── adversarial: verbose-but-wrong vs terse-but-correct        │
│  └── gold agreement: judge vs keyword overlap heuristic         │
│                                                                 │
│  A/B comparison  →  ab_report.json                              │
│  └── v1 (basic prompt) vs v2 (rubric + anchors) → winner       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup & Run Instructions

### Prerequisites
- Python 3.11+
- OpenAI API key (used for both generator and judge)

### Environment variables
```
OPENAI_API_KEY=           # required
GENERATOR_MODEL=          # default: gpt-4o-mini
JUDGE_MODEL=              # default: gpt-4o
JUDGE_TEMPERATURE=        # default: 0.0
GENERATOR_TEMPERATURE=    # default: 0.0
```

### Install
Same environment as Problem 1 — no additional dependencies.

### Run full pipeline (suite in → all reports out)
```bash
python judge/run_pipeline.py
# produces:
#   judge_data/results/suite_report.json
#   judge_data/results/bias_report.json
#   judge_data/results/validation_report.json
#   judge_data/results/ab_report.json
#   judge_data/results/audit_log.jsonl
```

### Run A/B comparison only
```bash
python judge/compare.py
```

### Judge and generator configured independently
Generator and judge are set via separate env vars (`GENERATOR_MODEL`, `JUDGE_MODEL`) and use separate client calls. Swapping either requires only a `.env` change — no code changes.

---

## Judging Mode & Rubric

**Mode: reference-based pointwise scoring.** Each test case has a gold answer (`expected_output`). The judge scores the generated answer against that reference on five criteria. Chosen over pairwise because: (a) gold answers exist from Problem 1, (b) pointwise scales linearly with suite size, (c) pairwise requires O(n²) comparisons and introduces position bias by design.

### Rubric

| Criterion | Definition | Score anchors |
|-----------|-----------|---------------|
| Correctness | Are the facts accurate and free of errors? | 0.0 = wrong, 0.5 = half correct, 1.0 = fully accurate |
| Faithfulness | Are all claims grounded in the reference? | 0.0 = fabricated, 1.0 = fully supported |
| Completeness | Does the answer cover all key points? | 0.0 = missing everything, 1.0 = covers all points |
| Instruction-following | Does the answer follow the system prompt? | 0.0 = ignores format, 1.0 = follows exactly |
| Conciseness | Appropriately brief without losing substance? | 0.0 = severely verbose/terse, 1.0 = well-calibrated |

Few-shot scoring anchors (0.0, 0.3, 0.5, 0.7, 1.0) are embedded in the rubric prompt to combat score clustering. Every criterion requires a one-sentence rationale grounded in specific evidence.

---

## Evaluation Results

### Suite Results (20 questions)

| Metric | Score |
|--------|-------|
| Pass rate (overall ≥ 0.7) | 0.95 |
| Mean correctness | 0.97 |
| Mean faithfulness | 0.83 |
| Mean completeness | 0.93 |
| Mean instruction-following | 0.935 |
| Mean conciseness | 0.725 |
| Mean overall | 0.878 |
| Total judge tokens | 20,349 |

One case failed (case 13 — self-attention, overall=0.58). The judge found the generated answer insufficiently grounded in the reference despite being factually reasonable — an honest finding rather than an artefact.

---

## Bias Handling

### Position Bias
**What it is:** In pairwise judging, the judge tends to prefer whichever answer appears first.

**Mitigation:** For 5 questions, two answers were generated (temp=0 and temp=0.7). Each pair was judged in both orders (A→B and B→A). A flip is recorded when the winner changes with order.

**Result: flip rate = 0.6 (3/5).** A 60% flip rate is significant — the judge's verdict is highly sensitive to presentation order. Mitigation: always average both-order scores in production pairwise setups, or require agreement across orders before accepting a verdict.

---

### Verbosity Bias
**What it is:** The judge scores longer answers higher regardless of content quality.

**Mitigation:** A correct answer was padded with 80 words of irrelevant filler. Both versions judged independently.

| Case | Original score | Padded score | Biased? |
|------|---------------|--------------|---------|
| 1 | 0.72 | 0.62 | No |
| 2 | 0.88 | 0.70 | No |
| 3 | 0.88 | 0.74 | No |

**Verbosity bias rate: 0.0.** The rubric's explicit conciseness criterion and scoring anchors penalized padding in all three cases.

---

### Sycophancy Bias
**What it is:** The judge is fooled by confident-sounding but factually wrong answers.

**Mitigation:** Three confidently-wrong answers injected (e.g. "RAG stands for Randomized Answer Generation, developed by Google for image recognition").

| Case | Overall | Correctness | Fooled? |
|------|---------|-------------|---------|
| 1 | 0.30 | 0.0 | No |
| 2 | 0.40 | 0.0 | No |
| 3 | 0.30 | 0.0 | No |

**Sycophancy fool rate: 0.0.** Correctness=0.0 on all three despite authoritative phrasing.

---

### Self-Enhancement Bias
**What it is:** A judge from the same model family as the generator inflates scores.

**Mitigation applied:** `gpt-4o` (judge) vs `gpt-4o-mini` (generator). The stronger judge is demonstrably harsher: mean faithfulness 0.83 and conciseness 0.725, both below the 0.95 self-reported scores when gpt-4o-mini judged itself in Problem 1.

**Limitation:** Both models are OpenAI family. Cross-provider judging was not possible with available API access. In production, a different-provider judge is strongly recommended.

---

### Score Clustering
**What it is:** Judges avoid extremes and cluster scores around 0.7–0.8.

**Mitigation:** Few-shot anchors at 0.0/0.3/0.5/0.7/1.0 in the rubric. Measured via score spread (std dev across criteria) in the A/B comparison.

**Result:** v2 (with anchors) avg spread 0.103 vs v1 (without) 0.076 — 35% improvement in discrimination.

---

## Judge Validation

### Test-Retest Consistency

| Metric | Value |
|--------|-------|
| Flip rate (pass/fail) | 0.2 (1/5) |
| Avg score delta | 0.04 |

One borderline case (0.68 vs 0.72) flipped between runs at temp=0. Even at zero temperature, LLM APIs are not perfectly deterministic. Averaging 2–3 runs per case is recommended for production use.

### Adversarial Probes

| Probe type | Overall | Correctness | Judge correct? |
|-----------|---------|-------------|----------------|
| Verbose-but-wrong (case 1) | 0.06 | 0.0 | ✓ caught |
| Terse-but-correct (case 1) | 0.94 | 1.0 | ✓ passed |
| Verbose-but-wrong (case 6) | 0.16 | 0.0 | ✓ caught |
| Terse-but-correct (case 6) | 0.98 | 1.0 | ✓ passed |

**Adversarial accuracy: 1.0.**

### Gold Agreement

| Metric | Value |
|--------|-------|
| Agreement rate | 1.0 |
| Cohen's kappa | 1.0 |

**Caveat:** 25% keyword overlap is a lenient threshold. Perfect agreement reflects both a working judge and an easy corpus — not a claim of perfect calibration on harder tasks.

---

## A/B Comparison

| Config | Mean score | Pass rate | Avg spread | Case wins |
|--------|-----------|-----------|------------|-----------|
| v1 — basic prompt | 0.962 | 1.0 | 0.076 | 2 |
| v2 — rubric + anchors | 0.888 | 0.9 | 0.103 | 7 |

**Winner: v2** — higher average spread (0.103 vs 0.076), meaning per-criterion scores are more differentiated and informative. v1's near-uniform 0.96 scores indicate clustering, not genuine evaluation. v2's lower mean and one failed case reflect stricter, more honest assessment.

**Would this judge gate a release?** With guardrails — yes, for low-stakes filtering. Require agreement across both pairwise orders, run each case twice and average, keep a human reviewer for any case scoring 0.6–0.8. The 60% position bias flip rate means single-order pairwise verdicts should never gate a release alone.

---

## Design Decisions & Trade-offs

### Why reference-based pointwise?
Gold answers exist for all 20 questions. Pointwise scoring scales linearly and allows per-criterion analysis. Pairwise would require O(n²) comparisons and introduces position bias structurally — the very bias being measured separately.

### Robust JSON parsing
Three-strategy fallback: direct parse → strip markdown fences → extract first `{...}` block. If all fail, the case scores zero with a `parse_error` flag rather than crashing the pipeline. Every raw response is logged to `audit_log.jsonl` for replay and inspection.

### Why gpt-4o as judge vs gpt-4o-mini as generator?
The stronger model is harsher and less likely to mirror the generator's style. Faithfulness (0.83) and conciseness (0.725) from gpt-4o are noticeably lower than self-reported 0.95 scores when gpt-4o-mini judged itself — evidence the capability gap partially mitigates self-enhancement. Cross-family judging remains the gold standard.

### Which bias was most concerning?
Position bias at 60% flip rate. A judge whose verdict flips based on answer order rather than answer quality is fundamentally unreliable for pairwise comparisons. The mitigation is procedural — always run both orders and require agreement — rather than solvable through prompt engineering alone.

---

# Repository Structure

```
rag-assign-1/
├── src/
│   ├── config.py            # env-based configuration
│   ├── ingest.py            # ingestion pipeline
│   ├── retrieve.py          # top-k retrieval with metadata filter
│   ├── answer.py            # LLM answer generation with citations
│   └── api.py               # FastAPI HTTP endpoint
├── eval/
│   ├── questions.json       # 20 evaluation questions with gold answers
│   ├── run_eval.py          # evaluation harness (retrieval + answer metrics)
│   └── results/
│       └── eval_results.json
├── judge/
│   ├── config.py            # judge + generator model config
│   ├── judge.py             # core judging logic, rubric, verdict parsing
│   ├── bias.py              # position, verbosity, sycophancy bias checks
│   ├── validate.py          # test-retest, adversarial probes, gold agreement
│   ├── compare.py           # A/B comparison, score spread, winner declaration
│   └── run_pipeline.py      # full pipeline entry point
├── judge_data/
│   ├── test_suite.json      # 20 test cases with gold answers
│   ├── prompts/
│   │   ├── v1.txt           # basic judge prompt (Config A)
│   │   └── v2.txt           # rubric + anchors prompt (Config B)
│   └── results/
│       ├── suite_report.json
│       ├── bias_report.json
│       ├── validation_report.json
│       ├── ab_report.json
│       └── audit_log.jsonl
├── docs/                    # corpus (10 Wikipedia HTML articles)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```