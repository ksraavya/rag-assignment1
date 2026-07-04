from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")   # OpenAI
JUDGE_MODEL     = os.getenv("JUDGE_MODEL", "gemini-2.0-flash")  # Gemini

JUDGE_TEMPERATURE      = float(os.getenv("JUDGE_TEMPERATURE", 0.0))
GENERATOR_TEMPERATURE  = float(os.getenv("GENERATOR_TEMPERATURE", 0.0))