from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH    = os.getenv("CHROMA_PATH", "./chroma_store")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K          = int(os.getenv("TOP_K", 5))
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-004")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-2.5-flash")