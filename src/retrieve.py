import time
import sys
from pathlib import Path

import chromadb
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import OPENAI_API_KEY, CHROMA_PATH, EMBED_MODEL, TOP_K

client     = OpenAI(api_key=OPENAI_API_KEY)
chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection("rag_docs")


def embed_query(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def retrieve(question: str, k: int = TOP_K, source_filter: str = None) -> dict:
    t0    = time.time()
    vec   = embed_query(question)
    where = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[vec],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    latency_ms = round((time.time() - t0) * 1000, 2)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text"       : doc,
            "source"     : meta.get("source"),
            "filename"   : meta.get("filename"),
            "chunk_index": meta.get("chunk_index"),
            "distance"   : round(dist, 4)
        })

    return {"chunks": chunks, "latency_ms": latency_ms}