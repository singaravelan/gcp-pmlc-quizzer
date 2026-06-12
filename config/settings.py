from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

# AI Backend selection
AI_BACKEND: str = os.getenv("AI_BACKEND", "claude")  # "claude" | "ollama"

# Claude settings
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MODEL_FAST: str = os.getenv("CLAUDE_MODEL_FAST", "claude-haiku-4-5-20251001")

# Ollama settings
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Embeddings backend when AI_BACKEND=claude: "huggingface" | "ollama"
EMBED_BACKEND: str = os.getenv("EMBED_BACKEND", "huggingface")
HUGGINGFACE_EMBED_MODEL: str = "all-MiniLM-L6-v2"

# App settings
APP_TITLE: str = os.getenv("APP_TITLE", "GCP PMLE Quizzer")
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}

# RAG settings
RAG_CHUNK_SIZE: int = 1000
RAG_CHUNK_OVERLAP: int = 200
RAG_TOP_K: int = 5

# Web search settings
SEARCH_NUM_RESULTS: int = 5
SEARCH_FETCH_TOP_N: int = 3
SEARCH_PAGE_MAX_CHARS: int = 8000
SEARCH_DELAY_SECONDS: float = 0.5

# Question generation
BLOOM_LEVELS: dict[int, str] = {
    3: "Application",
    4: "Analysis",
    5: "Synthesis",
    6: "Evaluation",
}

# Link validation
LINK_VALIDATE_TIMEOUT: int = 5
LINK_VALIDATE_MAX_WORKERS: int = 8
