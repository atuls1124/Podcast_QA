"""
Central configuration for the Podcast Q&A Bot.

Reads environment variables (via python-dotenv) and exposes:
- File system paths
- Model names and hyperparameters
- The YouTube URL of the podcast to ingest
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if it exists.
load_dotenv()

# ---------------------------------------------------------------------------
# File system layout
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
AUDIO_DIR: Path = DATA_DIR / "audio"
TRANSCRIPT_DIR: Path = DATA_DIR / "transcript"
EMBEDDINGS_DIR: Path = BASE_DIR / "embeddings"

# Make sure all required directories exist when the module is imported.
for _dir in (DATA_DIR, AUDIO_DIR, TRANSCRIPT_DIR, EMBEDDINGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# Default YouTube URL points to "Elon Musk x Nikhil Kamath | People by WTF Ep. 16".
# Override by setting YOUTUBE_URL in a .env file.
YOUTUBE_URL: str = os.getenv(
    "YOUTUBE_URL",
    "https://www.youtube.com/watch?v=Rni7Fz7208c",
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Whisper model size: tiny | base | small | medium | large
# Larger = more accurate, slower, and uses more RAM/VRAM.
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# Sentence-Transformers model used for both indexing and querying.
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# LLM used for answer generation. Default is a free OpenRouter model.
# Browse the full catalogue at https://openrouter.ai/models (filter by :free).
# Recommended free options (availability / rate limits vary):
#   - openai/gpt-oss-20b:free           (good instruction following, modest size)
#   - openai/gpt-oss-120b:free          (strongest free OpenAI-OSS model)
#   - meta-llama/llama-3.3-70b-instruct:free
#   - qwen/qwen3-next-80b-a3b-instruct:free
#   - z-ai/glm-4.5-air:free
LLM_MODEL: str = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b:free",
)
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "4"))

# Max characters per chunk. Whisper segments are typically short, so we merge
# a few of them together to give the retriever richer context.
CHUNK_MAX_CHARS: int = int(os.getenv("CHUNK_MAX_CHARS", "800"))

# ---------------------------------------------------------------------------
# OpenRouter API (OpenAI-compatible)
# ---------------------------------------------------------------------------
# Get a key at https://openrouter.ai/keys
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

# OpenRouter's OpenAI-compatible base URL. Override only if you proxy
# the request through another service.
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)

# App identity sent in the HTTP-Referer / X-Title headers (OpenRouter
# uses these for rankings and to identify the calling app).
APP_URL: str = os.getenv("APP_URL", "http://localhost:8501")
APP_NAME: str = os.getenv("APP_NAME", "Podcast Q&A Bot")

# ---------------------------------------------------------------------------
# Persisted artefacts
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH: Path = EMBEDDINGS_DIR / "faiss.index"
METADATA_PATH: Path = EMBEDDINGS_DIR / "metadata.pkl"
