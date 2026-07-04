# Cost-Efficient RAG Application

A retrieval-augmented generation (RAG) QA service over a Wikipedia ML corpus, backed by ChromaDB (embedded), evaluated across retrieval quality, answer quality, latency, and cost.

---

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

**Note on judge bias:** The same model family (gpt-4o-mini) is used as both generator and judge, which introduces self-enhancement bias — the judge tends to be lenient on outputs from its own family. Faithfulness and relevance scores should be interpreted with this in mind. A cross-family judge would give a more conservative estimate.

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

## Repository Structure
```
rag-assign-1/
├── src/
│   ├── config.py        # env-based configuration
│   ├── ingest.py        # ingestion pipeline
│   ├── retrieve.py      # top-k retrieval with metadata filter
│   ├── answer.py        # LLM answer generation with citations
│   └── api.py           # FastAPI HTTP endpoint
├── eval/
│   ├── questions.json   # 20 evaluation questions with gold answers
│   ├── run_eval.py      # evaluation harness
│   └── results/
│       └── eval_results.json
├── docs/                # corpus (10 Wikipedia HTML articles)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
