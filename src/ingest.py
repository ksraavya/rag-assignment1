import hashlib
import sys
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import pypdf
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from config import OPENAI_API_KEY, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL

# ── clients ────────────────────────────────────────────────────────────────────
client     = OpenAI(api_key=OPENAI_API_KEY)
chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(name="rag_docs")
splitter   = RecursiveCharacterTextSplitter(
                 chunk_size=CHUNK_SIZE,
                 chunk_overlap=CHUNK_OVERLAP
             )

# ── helpers ────────────────────────────────────────────────────────────────────
def chunk_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in resp.data]

# ── loaders ────────────────────────────────────────────────────────────────────
def load_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_html(path: Path) -> str:
    soup = BeautifulSoup(
        path.read_text(encoding="utf-8", errors="ignore"), "html.parser"
    )
    # target only the main article body
    article = soup.find("div", {"id": "mw-content-text"})
    if not article:
        article = soup  # fallback
    for tag in article(["script", "style", "nav", "footer",
                         "header", "table", "sup", ".reflist",
                         ".navbox", ".sistersitebox"]):
        tag.decompose()
    return article.get_text(separator="\n", strip=True)

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".html":
        return load_html(path)
    elif ext in (".md", ".txt"):
        return load_text(path)
    else:
        print(f"  [skip] unsupported type: {path.name}")
        return ""

# ── core ───────────────────────────────────────────────────────────────────────
def ingest_file(path: Path):
    print(f"ingesting: {path.name}")
    raw = load_file(path)
    if not raw.strip():
        print(f"  [warn] empty content, skipping")
        return

    chunks = splitter.split_text(raw)
    print(f"  chunks: {len(chunks)}")

    ids       = [chunk_id(c) for c in chunks]
    metadatas = [{"source": path.stem, "filename": path.name, "chunk_index": i}
                 for i, c in enumerate(chunks)]

    # embed in batches of 100
    vectors = []
    for i in range(0, len(chunks), 100):
        vectors += embed(chunks[i:i+100])

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=vectors,
        metadatas=metadatas
    )
    print(f"  done — upserted {len(chunks)} chunks")

def ingest_dir(docs_dir: str = "../docs"):
    p     = Path(docs_dir)
    files = [f for f in p.iterdir() if f.is_file()]
    if not files:
        print("no files found in docs/")
        return
    for f in sorted(files):
        ingest_file(f)
    print(f"\ntotal vectors in collection: {collection.count()}")

# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs_path = sys.argv[1] if len(sys.argv) > 1 else "./docs"
    ingest_dir(docs_path)