import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from config import TOP_K
from answer import ask

app = FastAPI(title="RAG QA Service")


# ── request / response models ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    k: int = TOP_K
    source_filter: str | None = None


class ChunkInfo(BaseModel):
    source: str
    chunk_index: int
    distance: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: list[ChunkInfo]
    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int
    completion_tokens: int


# ── endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    result = ask(req.question, k=req.k, source_filter=req.source_filter)
    return QueryResponse(
        question        = result["question"],
        answer          = result["answer"],
        chunks_used     = [ChunkInfo(**c) for c in result["chunks_used"]],
        retrieval_ms    = result["retrieval_ms"],
        generation_ms   = result["generation_ms"],
        prompt_tokens   = result["prompt_tokens"],
        completion_tokens = result["completion_tokens"]
    )